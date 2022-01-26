from __future__ import annotations

import random
from abc import ABC, abstractmethod

from Hint import ColorHint
from Move import PlayMove, DiscardMove, HintColorMove, HintValueMove
from Card import CARD_COLORS, Card


class Player(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def step(self, proxy: "PlayerGameProxy"):
        pass

    def play_random_card(self, proxy: "PlayerGameProxy"):
        """Play a random card"""
        hand = proxy.see_hand(self)
        index = random.randint(0, len(hand) - 1)
        return PlayMove(index)

    def give_random_hint(self, proxy: "PlayerGameProxy"):
        """Give a random hint to another player"""
        # naive player: play a random card
        other = random.choice(
            [
                p
                for p in proxy.get_other_players()
                if any(len(hints) < 2 for _, hints in proxy.see_hand(p))
            ]
        )

        # get the hand of the other player
        other_hand = proxy.see_hand(other)
        other_hand = [(card, hints) for card, hints in other_hand if len(hints) < 2]

        # select one card that is not completely known
        card, hints = random.choice(other_hand)

        if not hints or any(isinstance(hint, ColorHint) for hint in hints):
            return HintValueMove(other.name, card.value)

        return HintColorMove(other.name, card.color)
    
    def get_playable_cards(self, proxy: "PlayerGameProxy"):
        board = proxy.see_board()

        playable = [Card(c.color, c.value + 1) for c in board if c.value < 5]
        # Card(color, 1) for colors that are not on the board yet
        playable += [Card(color, 1) for color in CARD_COLORS if not any(c.color == color for c in board)]
        return playable

    def hint_playable(self, proxy: "PlayerGameProxy"):
        board = proxy.see_board()
        # next playable card for each color
        playable = self.get_playable_cards(proxy)

        for other in proxy.get_other_players():
            hand = proxy.see_hand(other)

            card, hints = next(((c, hints) for c, hints in hand if c in playable and len(hints) < 2), (None, None))
            if card:
                if any(isinstance(hint, ColorHint) for hint in hints):
                    return HintValueMove(other.name, card.value)
                else:
                    return HintColorMove(other.name, card.color)
        
        return None

    def __eq__(self, __o: object) -> bool:
        # player with the same name are the same player
        return isinstance(__o, Player) and self.name == __o.name

    def __str__(self) -> str:
        return self.name


class TrainablePlayer(Player):

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def receive_reward(self, reward):
        pass

    @abstractmethod
    def train(self, proxy: "PlayerGameProxy", reward: float):
        pass


class ConstantAgent(Player):
    """Naive agent that always plays the first card in its hand"""

    def step(self, proxy: "PlayerGameProxy"):
        return PlayMove(0)


class RandomAgent(Player):
    """Naive agent that always play a random card"""

    def step(self, proxy: "PlayerGameProxy") -> PlayMove:
        return self.play_random_card(proxy)


class PrompterAgent(Player):
    """Naive agent that gives a random hint (if legal) or discards a card"""

    def step(self, proxy: "PlayerGameProxy"):
        if proxy.count_blue_tokens():
            return self.give_random_hint(proxy)

        # no blue token available -> discard a random card
        my_hand = proxy.see_hand(self)
        return DiscardMove(random.randint(0, len(my_hand) - 1))


class RiskAverseAgent(Player):
    """Play a card only if both the color and the value are known. Otherwise, discard a random card."""

    def step(self, proxy: "PlayerGameProxy"):
        my_hand = proxy.see_hand(self)

        # find the index of a card whose color and value are both kwnown
        card_index = next(
            (i for i, (card, _) in enumerate(my_hand) if card.color and card.value),
            None,
        )
        if card_index is not None:
            return PlayMove(card_index)

        return DiscardMove(random.randint(0, len(my_hand) - 1))


class NaiveAgent(Player):
    """Try the following actions (in order): play a fully known card, give a random hint, discard a random card"""

    def step(self, proxy: "PlayerGameProxy"):
        my_hand = proxy.see_hand(self)

        playable_cards = self.get_playable_cards(proxy)

        # find the index of a card whose color and value are both kwnown
        card_index = next(
            (i for i, (card, _) in enumerate(my_hand) 
            if (card.color and card.value) or (card.value == 1 and all(c.value == 1 for c in playable_cards))),
            None,
        )
        if card_index is not None:
            return PlayMove(card_index)

        if proxy.count_blue_tokens():
            return self.hint_playable(proxy) or self.give_random_hint(proxy)
            # return self.give_random_hint(proxy)

        return DiscardMove(random.randint(0, len(my_hand) - 1))

