from typing import List, Set, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from Hint import ColorHint, Hint, ValueHint
from Card import Card, CARD_VALUES, CARD_COLORS
from Player import Player
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
#   ...                     |=> 25
#   hint 5 to player 5      -
#   hint red to player 1    -
#   ...                     |=> 25
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
    ) -> None:
        n_player = len(players)

        # 1. compute the player state

        self.players_state = torch.zeros(
            (n_player * 2, len(CARD_COLORS), len(CARD_VALUES))
        )

        for player_idx, player_state in enumerate(players):
            for card, hints in player_state:
                if card.color and card.value:
                    color_idx = CARD_COLORS.index(card.color)
                    value_idx = CARD_VALUES.index(card.value)

                    self.players_state[2 * player_idx, color_idx, value_idx] = 1

                for hint in hints:
                    if isinstance(hint, ColorHint):
                        color_idx = CARD_COLORS.index(hint.color)
                        self.players_state[2 * player_idx + 1, color_idx, :] = 1
                    elif isinstance(hint, ValueHint):
                        value_idx = CARD_VALUES.index(card.value)
                        self.players_state[2 * player_idx + 1, :, value_idx] = 1

        # 2. compute the board state

        self.board_state = torch.zeros((len(CARD_COLORS), len(CARD_VALUES)))

        for card in board:
            color_idx = CARD_COLORS.index(card.color)
            value_idx = CARD_VALUES.index(card.value)

            self.board_state[color_idx, value_idx] = 1

        # 3. compute the discard pile state

        self.discard_pile_state = torch.zeros((len(CARD_COLORS), len(CARD_VALUES)))

        for card in discard_pile:
            color_idx = CARD_COLORS.index(card.color)
            value_idx = CARD_VALUES.index(card.value)

            self.discard_pile_state[color_idx, value_idx] = 1

    def get_state(self):
        return self.players_state, self.board_state, self.discard_pile_state


class ActionDecoder:
    def __init__(self, outputs: torch.tensor) -> None:
        assert outputs.size()[0] == 60
        self.outputs = outputs

    def get_action(self):
        _, action_idx = torch.max(self.outputs, 0)
        idx = int(action_idx)

        if idx < 5:
            return PlayMove(idx), action_idx

        idx = idx - 5
        if idx < 25:
            return HintValueMove(idx // 5, CARD_VALUES[idx % 5]), action_idx

        idx = idx - 25
        if idx < 25:
            return HintColorMove(idx // 5, CARD_COLORS[idx % 5]), action_idx

        idx = idx - 25
        return DiscardMove(idx), action_idx


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # players have an initial dimension which is [bs, 2 * n. players, 5, 5]
        self.players_conv1 = nn.Conv2d(10, 32, 1, groups=2)
        self.players_conv2 = nn.Conv2d(32, 32, 1)

        self.full_conv1 = nn.Conv2d(34, 64, 1)
        self.full_conv2 = nn.Conv2d(64, 64 * 5 * 5, (5, 5))

        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 60)

    def forward(self, players_state, board_state, discard_pile_state):
        players_state = torch.unsqueeze(players_state, 0)
        x = F.relu(self.players_conv1(players_state))
        x = F.relu(self.players_conv2(x))

        x = x.squeeze()
        board_state = board_state.reshape((1, 5, 5)) #torch.unsqueeze(board_state, 0)
        discard_pile_state = discard_pile_state.reshape((1, 5, 5)) # torch.unsqueeze(discard_pile_state, 0)
        # x = torch.hstack([x, board_state, discard_pile_state])
        x = torch.cat([x, board_state, discard_pile_state], 0)
        x = x.unsqueeze(0)

        x = self.full_conv1(x)
        x = self.full_conv2(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x).flatten()

        return Q, F.softmax(Q)


class DRLAgent(Player):
    def __init__(self, name: str, discount=0.98) -> None:
        super().__init__(name)
        self.model = Net()

        self.discount = discount
        self.Qs = []
        self.actions = []

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)

    def prepare(self):
        self.Qs = []
        self.actions = []
        self.optimizer.zero_grad()

    def step(self, proxy: "PlayerGameProxy"):
        players_state = [proxy.see_hand(p) for p in [proxy.get_player(), *proxy.get_other_players()]]
        board_state = proxy.see_board()
        discard_pile_state = proxy.see_discard_pile()

        encoded_state = StateEncoder(players_state, board_state, discard_pile_state)

        Q, probs = self.model(*encoded_state.get_state())
        
        action, action_idx = ActionDecoder(probs).get_action()
        if isinstance(action, HintMove):
            action.player = [proxy.get_player(), *proxy.get_other_players()][action.player].name

        # log the state
        # self.states.append(deepcopy(encoded_state))
        self.Qs.append(Q)
        self.actions.append(action_idx)

        return action

    def train(self, reward):
        if len(self.Qs) == 0:
            return

        discounted_rewards = [reward * (self.discount ** i) for i in range(len(self.Qs))][::-1]

        # q values for this run
        # Q = torch.zeros((len(self.states), 60))
        targets = torch.zeros((len(self.Qs), 60))
        for i, (idx, reward) in enumerate(zip(self.actions, discounted_rewards)):
            # Q[i, idx] = reward
            targets[i, idx] = reward

        loss = F.mse_loss(torch.cat([q.unsqueeze(0) for q in self.Qs]), targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()