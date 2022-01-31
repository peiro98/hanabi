import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Move import HintMove
from Player import TrainablePlayer
from actions_decoder import ActionDecoder
from state_encoder import FlatStateEncoder

class Net(nn.Module):
    def __init__(self, n_players=5):
        super(Net, self).__init__()

        players_input_size = 100 * n_players  # [40, 60, 80, 100]
        output_size = 10 + 10 * (n_players - 1)

        input_size = players_input_size + 5 + 25 + 2  # player size + board + discard pile
        self.fc1 = nn.Linear(input_size, input_size * 4)
        self.fc2 = nn.Linear(input_size * 4, input_size * 4)
        self.fc3 = nn.Linear(input_size * 4, input_size * 2)
        self.fc4 = nn.Linear(input_size * 2, output_size * 2)
        self.fc5 = nn.Linear(output_size * 2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return self.fc5(x)


class DRLAgent(TrainablePlayer):
    def __init__(
        self,
        name: str,
        n_players=5,
        discount=0.95,
        training=True,
        eps=1.0,
        eps_step=0.999,
        finetune_eps=0.2,
        finetune_eps_step=0.9999,
        target_model_refresh_interval=10
    ) -> None:
        super().__init__(name)
        self.n_players = n_players

        self.model = Net(n_players)
        self.frozen_model = Net(n_players)

        self.discount = discount
        self.training = training

        self.played_games = 0
        self.experience = []
        self.target_model_refresh_interval = target_model_refresh_interval

        self.eps = eps
        self.eps_dict = defaultdict(lambda: self.eps)
        self.eps_step = eps_step
        self.finetune_eps = finetune_eps
        self.finetune_eps_step = finetune_eps_step
        self.min_eps = 0.1

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5e4, gamma=0.5)

    def prepare(self):
        """Reset the state of the player

        This method must be called before starting a new game.
        """
        self.states = []
        self.rewards = []
        self.actions = []

        self.model.eval()

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

    def step(self, proxy: "PlayerGameProxy"):
        encoded_state = self.__get_encoded_state(proxy)
        encoded_state = torch.from_numpy(encoded_state).float()

        # compute the output of the model
        # the Q array returned may require gradient as a result of the network computation
        # Therefore, .detach() allows to skip gradient computation
        Q = self.model(encoded_state).squeeze().detach()
        # create a decoder that is used to convert Q into actual actions
        decoder = ActionDecoder(Q.numpy())

        action = None
        is_random_action = random.random() < self.eps_dict[proxy.get_turn_index()]
        while (
            action is None
            # hints are not available
            # or (isinstance(action, HintMove) and proxy.count_blue_tokens() <= 0)
            # do not play or discard if this is the game's first turn
            # or ((isinstance(action, PlayMove) or isinstance(action, DiscardMove)) and proxy.get_turn_index() == 0)
        ):
            # generate a random number in [0, 1] to decide whether the player
            # is going to act greedily or not
            if is_random_action or action is not None:
                # act greedily
                action, action_idx = decoder.pick_random()
            else:
                # pick the best action
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

    def train(self, proxy: "PlayerGameProxy"):
        if len(self.states) == 0 or not self.training:
            return
        
        # state, action, reward, new_state
        for SARS in zip(self.states, self.actions, self.rewards, self.states[1:] + [None]):
            self.experience.append(SARS)

        for i in range(proxy.get_turn_index()):
            self.eps_dict[i] = max(self.min_eps, self.eps_dict[i] * self.eps_step)

        self.frozen_model.eval()
        self.model.train()
        self.optimizer.zero_grad()

        batch = list(random.sample(self.experience, min(64, len(self.experience))))

        state_shape = batch[0][0].shape

        # compute Q targets using the frozen model
        non_terminal_selector = torch.tensor([ns is not None for _, _, _, ns in batch], dtype=torch.int)

        next_states = torch.cat([ns if ns is not None else torch.zeros(state_shape) for _, _, _, ns in batch])
        target_Q = self.frozen_model(next_states)

        states = torch.cat([s for s, _, _, _ in batch])
        train_Q = self.model(states)
        # compute the target actions from the train Q
        _, actions_target = torch.max(train_Q, 1)

        target_Q = torch.sum(target_Q * torch.nn.functional.one_hot(actions_target, num_classes=20), 1)
        rewards = torch.tensor([r for _, _, r, _ in batch])
        target_Q = rewards + self.discount * non_terminal_selector * target_Q

        actions = torch.tensor([a for _, a, _, _ in batch])
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=(10 * (self.n_players)))
        train_Q = torch.sum(train_Q * actions_one_hot, 1)

        loss = F.mse_loss(train_Q, target_Q)
        if torch.isnan(loss):
            exit(1)

        loss.backward()
        print(loss.item())
        self.optimizer.step()
        self.scheduler.step()
        
        self.experience = self.experience[-(384*1024):]

        # increment the number of played games
        self.played_games += 1

    def save_pytorch_model(self, path: str):
        """Save the current PyTorch model"""
        torch.save(self.frozen_model.state_dict(), path)

    def save_numpy_model(self, path: str):
        """Save the current model as numpy arrays"""
        with open(path, 'wb') as f:
            # values are save as fc1.weight, fc1.bias, fc2.weight, fc2.bias, ...
            for value in self.frozen_model.state_dict().values():
                np.save(f, value.numpy())
