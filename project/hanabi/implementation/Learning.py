from cmath import exp
import itertools
from multiprocessing.sharedctypes import Value
from operator import itemgetter
from os import stat
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
        n_player = len(players)

        # 1. compute the player state

        # self.players_state = torch.zeros((n_player * 10, len(CARD_COLORS), len(CARD_VALUES)))
        ps = 5 * 20 * len(players)
        self.players_state = torch.zeros((ps))

        for player_idx, player_state in enumerate(players):
            for card_idx, (card, hints) in enumerate(player_state):
                if card.color and card.value:
                    color_idx = CARD_COLORS.index(card.color)
                    self.players_state[100 * player_idx + 20 * card_idx + color_idx] = 1

                if card.value:
                    value_idx = CARD_VALUES.index(card.value)
                    self.players_state[100 * player_idx + 20 * card_idx + 5 + value_idx] = 1

                for i, hint in enumerate(hints):
                    if isinstance(hint, ColorHint):
                        color_idx = CARD_COLORS.index(hint.color)
                        self.players_state[100 * player_idx + 20 * card_idx + 10 + i * 5 + color_idx] = 1
                    elif isinstance(hint, ValueHint):
                        value_idx = CARD_VALUES.index(card.value)
                        self.players_state[100 * player_idx + 20 * card_idx + 10 + i * 5 + value_idx] = 1

        # 2. compute the board state

        self.board_state = torch.zeros((len(CARD_COLORS), len(CARD_VALUES)))

        for card in board:
            color_idx = CARD_COLORS.index(card.color)
            value_idx = CARD_VALUES.index(card.value)

            self.board_state[color_idx, value_idx] = 1

        self.board_state = self.board_state.flatten()

        # 3. compute the discard pile state

        self.discard_pile_state = torch.zeros((len(CARD_COLORS), len(CARD_VALUES)))

        for card in discard_pile:
            color_idx = CARD_COLORS.index(card.color)
            value_idx = CARD_VALUES.index(card.value)

            self.discard_pile_state[color_idx, value_idx] = 1

        self.discard_pile_state = self.discard_pile_state.flatten()

        self.blue_tokens = blue_tokens
        self.red_tokens = red_tokens

    def get_state(self):
        return self.players_state, self.board_state, self.discard_pile_state, self.blue_tokens, self.red_tokens


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

        input_size = players_input_size + 25 + 25 + 2  # player size + board + discard pile
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size * 2)
        self.fc4 = nn.Linear(output_size * 2, output_size)

    def forward(self, players_state, board_state, discard_pile_state, blue_tokens, red_tokens):
        x = torch.tensor([blue_tokens, red_tokens])
        x = torch.cat([players_state, board_state, discard_pile_state, x])
        x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        Q = F.relu(self.fc4(x))

        return Q, F.softmax(Q, dim=0)


class DRLAgent(TrainablePlayer):
    def __init__(
        self,
        name: str,
        n_players=5,
        discount=0.95,
        training=True,
        eps=1.0,
        eps_step=0.999,
        target_model_refresh_interval=500,
    ) -> None:
        super().__init__(name)
        self.n_players = n_players

        self.model = Net(n_players)
        self.frozen_model = Net(n_players)

        self.discount = discount
        self.training = training

        self.played_games = 0
        self.positive_experience = []
        self.zero_experience = []
        self.target_model_refresh_interval = target_model_refresh_interval

        self.eps = eps
        self.eps_step = eps_step
        self.min_eps = 0.005

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def prepare(self):
        """Reset the state of the player

        This method must be called before starting a new game.
        """
        self.states = []
        self.rewards = []
        self.actions = []

        self.model.eval()

        self.eps = max(self.min_eps, self.eps * self.eps_step)

        # periodically the target model is refreshed
        if self.played_games % self.target_model_refresh_interval:
            self.frozen_model.load_state_dict(self.model.state_dict())

    def finetune(self, eps=0.1):
        self.eps_step = self.eps_step * 0.1
        self.eps = eps

        for g in self.optimizer.param_groups:
            g['lr'] *= 0.1

    def __get_encoded_state(self, proxy: "PlayerGameProxy") -> StateEncoder:
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
        )

    def step(self, proxy: "PlayerGameProxy"):
        encoded_state = self.__get_encoded_state(proxy)

        # compute the output of the model
        Q, _ = self.model(*encoded_state.get_state())
        # create a decoder that is used to convert Q into actual actions
        decoder = ActionDecoder(Q.squeeze())

        action = None
        is_random_action = random.random() < self.eps
        while (
            action is None
            # hints are not available
            or (isinstance(action, HintMove) and proxy.count_blue_tokens() <= 0)
            # do not play or discard if this is the game's first turn
            or ((isinstance(action, PlayMove) or isinstance(action, DiscardMove)) and proxy.get_turn_index() == 0)
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
        self.states.append(deepcopy(encoded_state))
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

    def train(self):
        if len(self.states) == 0 or not self.training:
            return

        # state, action, reward, new_state
        for SARS in zip(self.states, self.actions, self.rewards, self.states[1:] + [None]):
            if SARS[2] > 0:
                self.positive_experience.append(SARS)
            else:
                self.zero_experience.append(SARS)

        bs_positive = min(32, len(self.positive_experience))
        bs_zero = min(32, len(self.zero_experience))

        self.frozen_model.eval()

        input_states = []
        target_Q = []
        actions_idxs = []

        batch = itertools.chain(
            random.sample(self.positive_experience, bs_positive // 2),
            random.sample(self.zero_experience, bs_zero // 2),
        )
        for state, action, reward, next_state in batch:
            if next_state is not None:
                # compute the Q value for the next state
                Q, probs = self.frozen_model(*next_state.get_state())
                # compute the index of the best Q value
                Q = Q.squeeze()
                probs = probs.squeeze()
                _, best_Q_idx = torch.max(probs, 0)
                # take the best Q value
                best_Q = Q[best_Q_idx]
            else:
                best_Q = torch.tensor(0)

            # compute the target Q
            # size = 10 + 10 * (self.n_players - 1)
            # tq = torch.nn.functional.one_hot(torch.tensor(action), num_classes=size)
            # tq = tq * (reward + self.discount * best_Q)
            target_Q.append(reward + self.discount * best_Q)

            actions_idxs.append(action)
            input_states.append(state)

        self.model.train()
        self.optimizer.zero_grad()

        target_Q = torch.tensor(target_Q)
        target_Q.requires_grad = False

        outputs = torch.cat(
            [
                self.model(*state.get_state())[0].squeeze()[action].unsqueeze(0)
                for state, action in zip(input_states, actions_idxs)
            ]
        )

        loss = F.mse_loss(outputs, target_Q)
        if torch.isnan(loss):
            exit(1)

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        # self.positive_experience = list(
        #     random.sample(self.positive_experience, min(8192, len(self.positive_experience)))
        # )
        self.positive_experience = self.positive_experience[-8192:]
        self.zero_experience += list(random.sample(self.zero_experience, min(4096, len(self.zero_experience))))

        # increment the number of played games
        self.played_games += 1
