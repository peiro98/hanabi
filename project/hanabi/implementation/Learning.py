from cmath import exp
from collections import defaultdict
import itertools
from typing import List, Set, Tuple
from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from Hint import ColorHint, Hint, ValueHint
from Card import Card, CARD_VALUES, CARD_COLORS
from Player import TrainablePlayer
from Move import (
    DiscardMove,
    HintColorMove,
    HintMove,
    HintValueMove,
    PlayMove,
)

# n. of players -> n. of channels in the input
# (n. players x 2) x 5 x 5 = [100, 250] parameters
# [player 1, color, value] = True|False if the card is known or not
# [player 1 - hints, color, value] = True|False if the hint is given or not
# [player 2, color, value]
# [player 2 - hints, color, value]

# for the board
# 5 x 5 = 25 parameters
# [color, value] = 1 if played

# for the discard pile
# 5 x 5 = 25 parameters
# [color, value] = 1 if discarded

# [
#   play card 1             -
#   ...                     |=> 5
#   play card 5             -
#   hint 1 to player 1      -
#   ...                     |=> 20
#   hint 5 to player 5      -
#   hint red to player 1    -
#   ...                     |=> 20
#   hint yellow to player 5 -
#   discard 1               -
#   ...                     |=> 5
#   discard 5               -
# ]


class StateEncoder:
    """Encode the game state in torch tensors"""

    def __init__(
        self,
        players: List[List[Tuple[Card, Set[Hint]]]],
        board: List[Card],
        discard_pile: List[Card],
        blue_tokens: int,
        red_tokens: int,
    ) -> None:
        # 1. compute the player state

        # self.players_state = torch.zeros((n_player * 10, len(CARD_COLORS), len(CARD_VALUES)))
        ps = 5 * 20 * len(players)
        self.players_state = torch.zeros((ps))

        for player_idx, player_state in enumerate(players):
            for card_idx, (card, hints) in enumerate(player_state):
                if card.color:
                    color_idx = CARD_COLORS.index(card.color)
                    self.players_state[100 * player_idx + 20 * card_idx + color_idx] = 1

                if card.value:
                    value_idx = CARD_VALUES.index(card.value)
                    self.players_state[100 * player_idx + 20 * card_idx + 5 + value_idx] = 1

                for i, hint in enumerate(hints):
                    if isinstance(hint, ColorHint):
                        color_idx = CARD_COLORS.index(hint.color)
                        self.players_state[100 * player_idx + 20 * card_idx + 10 + color_idx] = 1
                    elif isinstance(hint, ValueHint):
                        value_idx = CARD_VALUES.index(card.value)
                        self.players_state[100 * player_idx + 20 * card_idx + 10 + 5 + value_idx] = 1

        # 2. compute the board state

        self.board_state = torch.zeros((len(CARD_COLORS)))

        for card in board:
            color_idx = CARD_COLORS.index(card.color)

            self.board_state[color_idx] = max(self.board_state[color_idx], card.value)

        self.board_state = self.board_state.flatten()

        # 3. compute the discard pile state

        self.discard_pile_state = torch.zeros((len(CARD_COLORS), len(CARD_VALUES)))

        for card in discard_pile:
            color_idx = CARD_COLORS.index(card.color)
            value_idx = CARD_VALUES.index(card.value)

            self.discard_pile_state[color_idx, value_idx] += 1

        self.discard_pile_state = self.discard_pile_state.flatten()

        self.blue_tokens = blue_tokens
        self.red_tokens = red_tokens

    def get_state(self):
        # return self.players_state, self.board_state, self.discard_pile_state, self.blue_tokens, self.red_tokens
        state = torch.tensor([self.blue_tokens, self.red_tokens])
        state = torch.cat([self.players_state, self.board_state, self.discard_pile_state, state])
        return state.unsqueeze(0)


class ActionDecoder:
    """Decode an action from a Q array"""

    def __init__(self, Q: torch.tensor) -> None:
        """
        Initialize the action decoder.

        Parameters:
        -----------
        Q: torch.tensor
            output of the Q-learning model
        """
        self.Q = Q

    def size(self):
        return self.Q.size()[0]

    def pick_action(self, mode="prob"):
        """Pick an action

        Parameters:
        -----------
        mode: str
            selection mode for the action ("prob" or "max")
        """
        if mode not in ["prob", "max"]:
            raise ValueError("Valid values for the mode parameter: ['max', 'prob']")

        if mode == "max":
            _, action_idx = torch.max(self.Q, 0)
        else:
            # transform Q values into probabilities
            probs = torch.clone(self.Q)
            # do not consider values below 1e-5
            probs[probs < 1e-5] = 0.0
            probs[probs > 0] = F.softmax(probs[probs > 0])
            # pick an action from the categorical distribution
            action_idx = torch.distributions.Categorical(probs).sample()

        return self.get(action_idx), action_idx

    def pick_random(self):
        n_players = (self.size() - 10) // 5

        c = random.choice(range(3))
        if c == 0:
            # random move
            action_idx = random.choice(range(5))
        elif c == 1:
            # random hint
            action_idx = random.randint(5, 5 + 10 * (n_players - 1) - 1)
        else:
            action_idx = random.randint(5 + 10 * (n_players - 1) - 1, 5 + 10 * (n_players - 1) - 1 + 5)

        return self.get(action_idx), action_idx

    def get(self, idx):
        n_players = (self.size() - 10) // 5

        if idx < 5:
            return PlayMove(idx)

        idx = idx - 5
        if idx < 5 * (n_players - 1):
            return HintValueMove(idx // 5, CARD_VALUES[idx % 5])

        idx = idx - 5 * (n_players - 1)
        if idx < 5 * (n_players - 1):
            return HintColorMove(idx // 5, CARD_COLORS[idx % 5])

        idx = idx - 5 * (n_players - 1)
        return DiscardMove(idx)


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
        Q = F.relu(self.fc5(x))

        return Q


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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
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
        if self.played_games and (self.played_games % self.target_model_refresh_interval) == 0:
            self.frozen_model.load_state_dict(self.model.state_dict())

    def __get_encoded_state(self, proxy: "PlayerGameProxy") -> torch.tensor:
        """Encode the current game state"""
        players_state = [proxy.see_hand(p) for p in [proxy.get_player(), *proxy.get_other_players()]]
        board_state = proxy.see_board()
        discard_pile_state = proxy.see_discard_pile()

        return StateEncoder(
            players_state,
            board_state,
            discard_pile_state,
            proxy.count_blue_tokens(),
            proxy.count_red_tokens(),
        ).get_state()

    def step(self, proxy: "PlayerGameProxy"):
        encoded_state = self.__get_encoded_state(proxy)

        # compute the output of the model
        Q = self.model(encoded_state)
        # create a decoder that is used to convert Q into actual actions
        decoder = ActionDecoder(Q.squeeze())

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

        if len(self.experience) < 64 * 1024:
            print(f"not yet ({len(self.experience)})")
            return

        for i in range(proxy.get_turn_index()):
            self.eps_dict[i] *= max(self.min_eps, self.eps_dict[i] * self.eps_step)


        self.frozen_model.eval()
        self.model.train()
        self.optimizer.zero_grad()

        batch = list(random.sample(self.experience, 32))

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
        
        self.experience = self.experience[-(256*1024):]

        # increment the number of played games
        self.played_games += 1
