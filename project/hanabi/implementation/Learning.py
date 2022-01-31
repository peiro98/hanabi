import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Move import HintMove
from Player import TrainablePlayer
from actions_decoder import ActionDecoder
from replay_memory import ReplayMemory, UniformReplayMemory
from state_encoder import FlatStateEncoder


# See https://stackoverflow.com/a/39757388
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HanabiGame import PlayerGameProxy


class MLPNetwork(nn.Module):
    """5-layers MLP

    The simple architecture of this network (only linear and ReLU layers)
    allows to easily port an equivalent inference-only model to numpy.
    """

    def __init__(self, n_players=2, amp_factor=4):
        """Initialize the MLP network

        Parameters
        ----------
        n_players : int
            number of players in the game, by default 2
        amp_factor : int, optional
            amplification factor of the input size to produce the hidden layers, by default 4
        """
        super(MLPNetwork, self).__init__()

        # each player is encoded through 100 values (see StateEncoder class)
        players_input_size = 100 * n_players
        # actions are one-hot encoded as described in the ActionDecoder class
        output_size = 10 + 10 * (n_players - 1)

        # total size = players sizes + board state size + discard pile state + num. of blue|red tokens
        input_size = players_input_size + 5 + 25 + 2

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * amp_factor),
            nn.ReLU(),
            nn.Linear(input_size * amp_factor, input_size * amp_factor),
            nn.ReLU(),
            nn.Linear(input_size * amp_factor, input_size * (amp_factor // 2)),
            nn.ReLU(),
            nn.Linear(input_size * (amp_factor // 2), output_size * 2),
            nn.ReLU(),
            nn.Linear(output_size * 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            network's output
        """
        return self.net(x)


class DRLAgent(TrainablePlayer):
    """Double Q-Learning

    See [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
    """

    def __init__(
        self,
        name: str,
        network_builder=None,
        discount=0.95,
        training=True,
        initial_eps=1.000,
        eps_step=0.99995,
        minimum_eps=0.1,
        turn_dependent_eps=True,
        batch_size=64,
        target_model_refresh_interval=10,
        replay_memory: ReplayMemory = UniformReplayMemory(384 * 1024),
    ) -> None:
        super().__init__(name)

        if network_builder is None:
            self.model = MLPNetwork()
            self.frozen_model = MLPNetwork()
        else:
            self.model = network_builder()
            self.frozen_model = network_builder()

        self.discount = discount
        self.training = training
        self.batch_size = batch_size

        self.played_games = 0
        self.replay_memory = replay_memory
        self.target_model_refresh_interval = target_model_refresh_interval

        self.eps_step = eps_step
        self.minimum_eps = minimum_eps
        if turn_dependent_eps:
            # The epsilon coefficient is determined according to the turn number.
            # At the end of the game, the epsilon values "used" during the game
            # are decreased by a factor self.eps_step
            self.eps = defaultdict(lambda: initial_eps)
        else:
            # There is only one global epsilon coefficient
            self.eps = initial_eps

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5e4, gamma=0.5)

    def prepare(self):
        """Reset the state of the player

        This method must be called before starting a new game.
        """

        # Clear the experience collected during a previous game
        self.states = []
        self.rewards = []
        self.actions = []

        # periodically the target model is refreshed
        if (self.played_games % self.target_model_refresh_interval) == 0:
            self.frozen_model.load_state_dict(self.model.state_dict())

    def __get_encoded_state(self, proxy: "PlayerGameProxy") -> torch.tensor:
        """Encode the current game state"""
        players_state = [proxy.see_hand(p) for p in [proxy.get_player(), *proxy.get_other_players()]]
        board_state = proxy.see_board()
        discard_pile_state = proxy.see_discard_pile()

        return FlatStateEncoder(
            players_state,
            board_state,
            discard_pile_state,
            proxy.count_blue_tokens(),
            proxy.count_red_tokens(),
        ).get()

    def step(self, proxy: PlayerGameProxy):
        # Extract the current state from the game's proxy
        encoded_state = self.__get_encoded_state(proxy)
        # Convert the encoded state to a torch float tensor
        encoded_state = torch.from_numpy(encoded_state).float()

        # Compute the output of the training model
        self.model.eval()  # Make sure the network is in evaluation mode
        with torch.no_grad():  # Do not generate gradients
            Q = self.model(encoded_state).squeeze()
        # Instantiate an action decoder for the network output
        decoder = ActionDecoder(Q.numpy())

        action = None

        is_random_action = False
        # Select the epsilon coefficient
        eps = self.eps if type(self.eps) == float else self.eps[proxy.get_turn_index()]
        # Generate a random number in [0, 1] to decide whether the player
        # is going to act greedily or not
        is_random_action = random.random() < eps

        if is_random_action:
            # pick a random action
            action, action_idx = decoder.pick_random()
        else:
            # act greedily
            # (mode='prob' appears to perform badly and/or the training is very slow)
            action, action_idx = decoder.pick_action(mode="max")

        # hint actions refer to players using their indices
        if isinstance(action, HintMove):
            # convert indices to names
            action.player = proxy.get_other_players()[action.player].name

        # log state, reward and selected action
        self.states.append(torch.clone(encoded_state))
        self.rewards.append(0)
        self.actions.append(action_idx)

        return action

    def receive_reward(self, reward: float):
        """Reward the last performed action during this game

        Parameters:
        -----------
        reward: float
            the reward
        """
        self.rewards[-1] += reward

    def __update_epsilon(self, turn_index):
        if type(self.eps) == float:
            self.eps = max(self.minimum_eps, self.eps * self.eps_step)
        else:
            for turn_index in range(turn_index):
                self.eps[turn_index] = max(self.minimum_eps, self.eps[turn_index] * self.eps_step)

    def train(self, proxy: PlayerGameProxy):
        if len(self.states) == 0 or not self.training:
            return

        # Add all the states encountered during this game to the replay memory
        self.replay_memory.add_experience_from_game(self.states, self.actions, self.rewards)

        # Update the epsilon value, possibly taking into account the number of turn completed
        # during this game
        self.__update_epsilon(proxy.get_turn_index())

        # Set the target model in evaluation mode
        self.frozen_model.eval()
        # and the training model in training mode
        self.model.train()

        # Zero the gradients of the training model
        self.optimizer.zero_grad()

        batch = self.replay_memory.sample(self.batch_size)

        # Convert the next states into an array of size (batch_size, state_size)
        next_states = torch.cat(
            [
                # next_state is None => its previous state was a terminal state
                next_state if next_state is not None else torch.zeros(batch[0][0].shape)
                for _, _, _, next_state in batch
            ]
        )

        ####################
        # Action selection #
        ####################

        # Select the best ACTION to perform in the NEXT STATE(S) according to the ONLINE NETWORK
        with torch.no_grad():
            next_Q_online = self.model(next_states)

        _, selected_actions = torch.max(next_Q_online, 1)
        # One hot encode the actions on axis 1
        selected_actions = torch.nn.functional.one_hot(selected_actions, num_classes=next_Q_online.shape[1])

        #####################
        # Action evaluation #
        #####################

        # Evaluate Q for the NEXT STATE(S) according to the TARGET NETWORK
        with torch.no_grad():
            next_Q_target = self.frozen_model(next_states)

        # For each action, take the corresponding Q value computed by the target model
        # and sum to the actual rewards obtained.
        rewards = torch.tensor([reward for _, _, reward, _ in batch])
        # The sum is used to obtain the only non-zero element on each row
        next_Q = rewards + self.discount * torch.sum(next_Q_target * selected_actions, 1)

        ########################
        # Current Q evaluation #
        ########################

        # Compute the Q values for the states using the training models
        states = torch.cat([state for state, _, _, _ in batch])
        current_Q = self.model(states)

        # One-hot encode the actual actions performed as part of the experience of the agent
        actions = torch.tensor([action for _, action, _, _ in batch])
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=current_Q.shape[1])
        # Compute the Q value for the pairs (current_state, performed_action)
        current_Q = torch.sum(current_Q * actions_one_hot, 1)

        ###################
        # Loss evaluation #
        ###################

        # The loss is computed as the the difference between:
        #  - the current estimate of Q for the inputs (state, action)
        #  - the sum of the actual reward and the Q value estimate by the target model
        #    for (next_state, next_action) where the next_action is selected by the
        #    training model.
        # Broadly speaking, reduce the distance between the decisions made by the model
        # and the evidencies from the experience.
        loss = F.mse_loss(current_Q, next_Q)

        if torch.isnan(loss):
            # something very bad just happened
            exit(1)

        # Compute the gradients from the loss
        loss.backward()
        print(loss.item())

        # Apply the optimizer step (recompute the parameters)
        self.optimizer.step()
        # and update the learning rate
        self.lr_scheduler.step()

        # Update the number of played games
        self.played_games += 1

    def save_pytorch_model(self, path: str):
        """Save the current PyTorch model"""
        torch.save(self.frozen_model.state_dict(), path)

    def save_numpy_model(self, path: str):
        """Save the current model as numpy arrays"""
        with open(path, "wb") as f:
            # values are save as fc1.weight, fc1.bias, fc2.weight, fc2.bias, ...
            for value in self.frozen_model.state_dict().values():
                np.save(f, value.numpy())
