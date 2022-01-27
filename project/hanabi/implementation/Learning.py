from cmath import exp
import itertools
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

eps = 0.9999

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
    def __init__(self, n_players, Q: torch.tensor, eps=1.0) -> None:
        self.n_players = n_players
        self.Q = Q
        self.eps = eps

    def get_action(self):
        _, action_idx = torch.max(self.Q, 0)

        if random.random() < eps:
            c = random.choice(range(3))
            if c == 0:
                # random move
                action_idx = random.choice(range(5))
            elif c == 1:
                # random hint
                action_idx = random.randint(5, 5 + 10 * (self.n_players - 1) - 1)
            else:
                action_idx = random.randint(5 + 10 * (self.n_players - 1) - 1, 5 + 10 * (self.n_players - 1) - 1 + 5)
        # if random.random() < self.eps:
        #     idx = random.randint(0, int(self.outputs.size()[0]) - 1)

        idx = int(action_idx)

        if idx < 5:
            return PlayMove(idx), action_idx

        idx = idx - 5
        if idx < 5 * (self.n_players - 1):
            return HintValueMove(idx // 5, CARD_VALUES[idx % 5]), action_idx

        idx = idx - 5 * (self.n_players - 1)
        if idx < 5 * (self.n_players - 1):
            return HintColorMove(idx // 5, CARD_COLORS[idx % 5]), action_idx

        idx = idx - 5 * (self.n_players - 1)
        return DiscardMove(idx), action_idx

class Net(nn.Module):
    def __init__(self, n_players=5):
        super(Net, self).__init__()

        players_input_size = 100 * n_players # [40, 60, 80, 100]
        output_size = 10 + 10 * (n_players - 1)

        input_size = players_input_size + 25 + 25 + 2 # player size + board + discard pile
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
    def __init__(self, name: str, n_players=5, discount=0.95, training=True, target_model_refresh_interval=500) -> None:
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

        self.optimizer = torch.optim.Adam(self.model.parameters())


    def prepare(self):
        """Reset the state of the player
        
        This method must be called before starting a new game.
        """
        self.states = []
        self.rewards = []
        self.Qs = []
        self.actions = []

        self.model.eval()

        # periodically the target model is refreshed
        if self.played_games % self.target_model_refresh_interval:
            self.frozen_model.load_state_dict(self.model.state_dict())


    def __get_encoded_state(self, proxy: "PlayerGameProxy"):
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
        global eps
        eps = eps * 0.9999

        Q, _ = self.model(*encoded_state.get_state())
        Q = Q.squeeze()

        action = None
        while action is None or (isinstance(action, HintMove) and proxy.count_blue_tokens() <= 0):
            action, action_idx = ActionDecoder(self.n_players, Q, eps).get_action()

        if isinstance(action, HintMove):
            action.player = proxy.get_other_players()[action.player].name

        # log the state
        # self.states.append(deepcopy(encoded_state))
        self.states.append(deepcopy(encoded_state))
        self.Qs.append(Q)
        self.rewards.append(0)
        self.actions.append(action_idx)

        return action

    def receive_reward(self, reward: float):
        self.rewards[-1] += reward

    def train(self):
        if len(self.Qs) == 0 or not self.training:
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

        batch = itertools.chain(random.sample(self.positive_experience, bs_positive // 2), random.sample(self.zero_experience, bs_zero // 2))
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

        outputs = torch.cat([
            self.model(*state.get_state())[0].squeeze()[action].unsqueeze(0)
            for state, action in zip(input_states, actions_idxs)
        ])
        
        loss = F.mse_loss(outputs, target_Q)

        loss.backward()
        self.optimizer.step()
            # self.scheduler.step()

        self.positive_experience = list(random.sample(self.positive_experience, min(4096, len(self.positive_experience))))
        # self.positive_experience = exp_with_reward[-8192:]
        self.zero_experience += list(random.sample(self.zero_experience, min(4096, len(self.zero_experience))))

        # increment the number of played games
        self.played_games += 1