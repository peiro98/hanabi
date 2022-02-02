from abc import ABC, abstractmethod
from typing import List, Tuple, Set

import numpy as np

from .cards import CARD_COLORS, CARD_VALUES, Card
from .hints import Hint, ColorHint, ValueHint


class StateEncoder(ABC):
    """Encode the game state in a numpy array"""

    @abstractmethod
    def __init__(
        self,
        players: List[List[Tuple[Card, Set[Hint]]]],
        board: List[Card],
        discard_pile: List[Card],
        blue_tokens: int,
        red_tokens: int,
    ) -> None:
        """Create an instance of the state encoder

        Parameters
        ----------
        players : List[List[Tuple[Card, Set[Hint]]]]
            list of players hands. Each player's hand is represented as List[Tuple[Card, Set[Hint]]].
        board : List[Card]
            list of card on the board
        discard_pile : List[Card]
            list of discarded cards
        blue_tokens : int
            number of *available* blue tokens
        red_tokens : int
            number of red token *left*
        """
        pass

    @abstractmethod
    def get(self) -> np.ndarray:
        """Get the encoded state as numpy array

        Returns
        -------
        np.ndarray
            the array encoding the state
        """
        pass

    def n_players(self) -> int:
        """Return the number of players for this encoded state"""
        raise NotImplementedError()


class FlatStateEncoder(StateEncoder):
    """Encode the game state in a *flat* (1-D) numpy array

    Each card encoded by 20 values:
     - one-hot encoding of the card's color (5 entries)
     - one-hot encoding of the card's value (5 values)
     - one-hot encoding of the card's color hint (5 values)
     - one-hot encoding of the card's value hint (5 values)

    Each player has a maximum of 5 cards, resulting in a total of
    100 values per player. An hanabi game hosts 2 to 5 players.
    Therefore, a total of 200 to 500 values encode the players' hands.

    The board is encoded as a 5 values array, each representing the
    maximum value played for a certain color.
    As an example, if the board contains the following cards
      [(1, R), (2, R), (1, B), (2, B), (3, B)]
    the corresponding encoded board state would be
        [2, 0, 0, 3, 0]
         ^  ^  ^  ^  ^
        [R, Y, G, B, W]

    An array counts the number of discarded cards by color and value.

    All this informations are flattened and concatenated with the number of
    available blue tokens and the number of remaining red tokens.
    """

    def __init__(
        self,
        players: List[List[Tuple[Card, Set[Hint]]]],
        board: List[Card],
        discard_pile: List[Card],
        blue_tokens: int,
        red_tokens: int,
    ) -> None:
        """Create an instance of the state encoder

        Parameters
        ----------
        players : List[List[Tuple[Card, Set[Hint]]]]
            list of players hands. Each player's hand is represented as List[Tuple[Card, Set[Hint]]].
        board : List[Card]
            list of card on the board
        discard_pile : List[Card]
            list of discarded cards
        blue_tokens : int
            number of *available* blue tokens
        red_tokens : int
            number of red token *left*
        """
        super(__class__, self).__init__(players, board, discard_pile, blue_tokens, red_tokens)

        # 1. Encode players' hands

        card_size = 20  # number of values required to encode a card and its hints
        hand_size = 5 * 20  # number of values required to encode an hand
        self.players_state = np.zeros((len(players) * hand_size))

        for hand_idx, hand in enumerate(players):
            # for each hand

            for card_idx, (card, hints) in enumerate(hand):
                # for each card in the hand

                # start index of the values encoding the card's state
                state_idx = hand_size * hand_idx + card_size * card_idx

                if card.color:
                    # color is known => set to one the corresponding entry
                    color_idx = CARD_COLORS.index(card.color)
                    # the entries [0..4] of each card's state encode the color
                    self.players_state[state_idx + color_idx] = 1

                if card.value:
                    # value is known => set to one the corresponding entry
                    value_idx = CARD_VALUES.index(card.value)
                    # the entries [5..9] of each card's state encode the color
                    self.players_state[state_idx + 5 + value_idx] = 1

                # assert(len(hints) <= 2)
                for hint in hints:
                    # for each hint provided for this card
                    factor = -1 if hint.negative else 1

                    if isinstance(hint, ColorHint):
                        color_idx = CARD_COLORS.index(hint.color)
                        # the entries [10..14] of each card's state encode the color
                        self.players_state[state_idx + 10 + color_idx] = 1 * factor

                    elif isinstance(hint, ValueHint):
                        # the entries [15..19] of each card's state encode the value
                        value_idx = CARD_VALUES.index(hint.value)
                        self.players_state[state_idx + 15 + value_idx] = 1 * factor

        # 2. Encode the board state

        self.board_state = np.zeros((len(CARD_COLORS)))

        for card in board:
            # for each card on the board
            color_idx = CARD_COLORS.index(card.color)
            # compute the maximum card available on the board for the card's color
            self.board_state[color_idx] = max(self.board_state[color_idx], card.value)

        # 3. Encode the discard pile state

        self.discard_pile_state = np.zeros((len(CARD_COLORS), len(CARD_VALUES)))

        for card in discard_pile:
            # for each card in the discard pile
            color_idx = CARD_COLORS.index(card.color)
            value_idx = CARD_VALUES.index(card.value)

            # increment the number of discarded cards with same color and value
            self.discard_pile_state[color_idx, value_idx] += 1

        self.discard_pile_state = self.discard_pile_state.flatten()

        # 4. Encode the number of blue and red tokens

        self.blue_tokens = blue_tokens
        # self.blue_tokens = 1 if blue_tokens > 0 else 0
        self.red_tokens = red_tokens

    def get(self):
        """Get the encoded state as numpy array

        Returns
        -------
        np.ndarray
            the array encoding the state
        """
        state = np.concatenate(
            [
                self.players_state,
                self.board_state,
                self.discard_pile_state,
                np.array([self.blue_tokens, self.red_tokens]),
            ]
        )
        return np.expand_dims(state, 0)
    
    def n_players(self) -> int:
        """Return the number of players for this encoded state"""
        return (self.get().size - 32) // 100
