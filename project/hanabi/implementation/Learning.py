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
        blue_tokens: int,
        red_tokens: int,
    ) -> None:
        n_player = len(players)

        # 1. compute the player state

        self.players_state = torch.zeros((n_player * 10, len(CARD_COLORS), len(CARD_VALUES)))

        for player_idx, player_state in enumerate(players):
            for card_idx, (card, hints) in enumerate(player_state):
                if card.color and card.value:
                    color_idx = CARD_COLORS.index(card.color)
                    value_idx = CARD_VALUES.index(card.value)

                    self.players_state[10 * player_idx + 2 * card_idx, color_idx, value_idx] = 1

                for hint in hints:
                    if isinstance(hint, ColorHint):
                        color_idx = CARD_COLORS.index(hint.color)
                        self.players_state[10 * player_idx + 2 * card_idx + 1, color_idx, :] = 1
                    elif isinstance(hint, ValueHint):
                        value_idx = CARD_VALUES.index(card.value)
                        self.players_state[10 * player_idx + 2 * card_idx + 1, :, value_idx] = 1

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

        self.blue_tokens = blue_tokens
        self.red_tokens = red_tokens

    def get_state(self):
        return self.players_state, self.board_state, self.discard_pile_state, self.blue_tokens, self.red_tokens


class ActionDecoder:
    def __init__(self, n_players, Q: torch.tensor, eps=1.0) -> None:
        assert Q.size()[0] == 10 + 10 * n_players
        self.n_players = n_players
        self.Q = Q
        self.eps = eps

    def get_action(self):
        _, action_idx = torch.max(self.Q, 0)

        if random.random() < 0.5:
            c = random.choice(range(3))
            if c == 0:
                # random move
                action_idx = random.choice(range(5))
            elif c == 1:
                # random hint
                action_idx = random.randint(5, 5 + 10 * self.n_players - 1)
            else:
                action_idx = random.randint(5 + 10 * self.n_players - 1, 5 + 10 * self.n_players - 1 + 5)
        # if random.random() < self.eps:
        #     idx = random.randint(0, int(self.outputs.size()[0]) - 1)

        idx = int(action_idx)

        if idx < 5:
            return PlayMove(idx), action_idx

        idx = idx - 5
        if idx < 5 * self.n_players:
            return HintValueMove(idx // 5, CARD_VALUES[idx % 5]), action_idx

        idx = idx - 5 * self.n_players
        if idx < 5 * self.n_players:
            return HintColorMove(idx // 5, CARD_COLORS[idx % 5]), action_idx

        idx = idx - 5 * self.n_players
        return DiscardMove(idx), action_idx


class Net(nn.Module):
    def __init__(self, n_players=5):
        super(Net, self).__init__()

        input_size = 10 * n_players

        # players have an initial dimension which is [bs, 2 * n. players, 5, 5]
        self.players_conv1 = nn.Conv2d(input_size, input_size * 2, 1, groups=n_players)
        self.players_conv2 = nn.Conv2d(input_size * 2, input_size, 1)

        # self.full_conv1 = nn.Conv2d(50 + 2, 64, 1)
        self.full_conv = nn.Conv2d(input_size + 2, (input_size + 2) * 5 * 5, (5, 5))

        self.fc1 = nn.Linear((input_size + 2) * 5 * 5 + 2, 192)
        self.fc2 = nn.Linear(192, 96)
        self.fc3 = nn.Linear(96, 10 + 10 * n_players)

    def forward(self, players_state, board_state, discard_pile_state, blue_tokens, red_tokens):
        players_state = torch.unsqueeze(players_state, 0)
        x = F.relu(self.players_conv1(players_state))
        x = F.relu(self.players_conv2(x))

        x = x.squeeze()
        board_state = board_state.reshape((1, 5, 5))  # torch.unsqueeze(board_state, 0)
        discard_pile_state = discard_pile_state.reshape((1, 5, 5))  # torch.unsqueeze(discard_pile_state, 0)
        # x = torch.hstack([x, board_state, discard_pile_state])
        x = torch.cat([x, board_state, discard_pile_state], 0)
        x = x.unsqueeze(0)

        x = F.relu(self.full_conv(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.hstack([x, torch.tensor([blue_tokens, red_tokens]).unsqueeze(0)])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x).flatten()

        return Q, F.softmax(Q, dim=0)


class DRLAgent(TrainablePlayer):
    def __init__(self, name: str, n_players=5, discount=0.95, training=True) -> None:
        super().__init__(name)
        self.n_players = n_players
        self.model = Net(n_players)
        self.discount = discount
        self.training = training

        self.experience = []

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 1)

    def prepare(self):
        """Reset the state of the player
        
        This method must be called before starting a new game.
        """
        self.states = []
        self.rewards = []
        self.Qs = []
        self.actions = []

        self.model.eval()

    def clear_illegal_moves(self, Q, proxy):
        Q[5:10] = 0
        Q[5 + 5 * self.n_players:5 + 5 * self.n_players + 5] = 0
        if proxy.count_blue_tokens() == 0:
            Q[5:5 + 10 * self.n_players] = 0
        return Q

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

    def step(self, proxy: "PlayerGameProxy", eps=1.0):
        encoded_state = self.__get_encoded_state(proxy)

        Q, _ = self.model(*encoded_state.get_state())
        Q = self.clear_illegal_moves(Q, proxy)

        action = None
        while action is None or (isinstance(action, HintMove) and (action.player == 0 or proxy.count_blue_tokens() <= 0)):
            action, action_idx = ActionDecoder(self.n_players, Q, eps).get_action()

        if isinstance(action, HintMove):
            action.player = [proxy.get_player(), *proxy.get_other_players()][action.player].name

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
        for SARS in zip(self.states[:-1], self.actions[:-1], self.rewards[:-1], self.states[1:]):
            self.experience.append(SARS)

        self.experience.append((self.states[-1], self.actions[-1], self.rewards[-1], None))

        batch_size = min(64, len(self.experience))

        input_states = []
        target_Q = []

        for state, action, reward, next_state in random.sample(self.experience, batch_size):
            if next_state is not None:
                # compute the Q value for the next state
                Q, probs = self.model(*next_state.get_state())
                # compute the index of the best Q value
                _, best_Q_idx = torch.max(probs, 0)
                # take the best Q value
                best_Q = Q[best_Q_idx]
            else:
                best_Q = 0

            # compute the target Q
            size = 10 + 10 * self.n_players
            tq = torch.nn.functional.one_hot(torch.tensor(action), num_classes=size)
            tq = tq * (reward + self.discount * best_Q)
            target_Q.append(tq)

            input_states.append(state)

        self.model.train()
        self.optimizer.zero_grad()

        outputs = torch.cat([self.model(*state.get_state())[0].unsqueeze(0) for state in input_states])
        target_Q = torch.cat([tq.unsqueeze(0) for tq in target_Q])
        
        loss = F.mse_loss(outputs, target_Q)

        loss.backward()
        print(loss.item())
        self.optimizer.step()
        # self.scheduler.step()
