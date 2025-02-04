from typing import List, Optional
from random import shuffle
from itertools import product

from .hints import ColorHint, Hint, ValueHint

# card colors
CARD_COLORS = ["R", "Y", "G", "B", "W"]
CARD_COLORS_EXTENDED = ["RED", "YELLOW", "GREEN", "BLUE", "WHITE"]

# number of cards for each value (1, 2, 3, 4, 5)
CARD_AVAILABILITY = [3, 2, 2, 2, 1]

# card values
CARD_VALUES = [1, 2, 3, 4, 5]


class Card:
    """A card.

    This class represents a card in an Hanabi game. The knowledge of
    a card's color and value depends on the card state (played, in-hand or
    discarded) and on the player (players have full knowledge of the
    hands of the other players and partial knowledge of their hands).
    Therefore, the color and value properties are defined as optional.

    """

    def __init__(self, color: Optional[str], value: Optional[int]):
        if color and color not in CARD_COLORS:
            raise ValueError("Invalid color!")
        if value and value not in CARD_VALUES:
            raise ValueError("Invalid value!")

        self.color = color
        self.value = value

    def mask(self, hints: List[Hint]):
        """Return a copy of this card, masked by the passed hints."""
        color, value = None, None
        for hint in hints:
            if isinstance(hint, ColorHint):
                color = hint.color
            elif isinstance(hint, ValueHint):
                value = hint.value
        return Card(color, value)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Card) and self.color == __o.color and self.value == __o.value

    def __str__(self) -> str:
        return f"({self.color or '-'}, {self.value or '-'})"

    def __repr__(self) -> str:
        return self.__str__()


def build_cards_deck() -> List[Card]:
    """Build an Hanabi card deck.

    The returned deck is NOT shuffled: cards are ordered by color and value
    """
    return [
        Card(color, value)
        for color, (value, n) in product(CARD_COLORS, zip(CARD_VALUES, CARD_AVAILABILITY))
        for _ in range(n)
    ]


class Deck:
    """A deck of cards"""

    def __init__(self) -> None:
        # load and shuffle the cards deck
        self.cards = build_cards_deck()
        shuffle(self.cards)

    def pick_card(self) -> Card:
        if not self.cards:
            raise ValueError("No more cards available")
        return self.cards.pop()

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def __str__(self) -> str:
        return f"Deck(cards left: {len(self.cards)})"

    def __repr__(self) -> str:
        return self.__str__()


class PredictableDeck(Deck):
    """A predictable deck of cards

    One of the players gets an hand full of 1s. Indeed, the first
    two hands are:
     - [(Y, 3), (R, 2), (B, 5), (Y, 2), (G, 4)]
     - [(W, 1), (B, 1), (R, 1), (G, 1), (Y, 1)]
    """

    def __init__(self) -> None:
        self.cards = [
            Card("Y", 3),
            Card("R", 2),
            Card("B", 5),
            Card("Y", 2),
            Card("G", 4),
            Card("W", 1),
            Card("B", 1),
            Card("R", 1),
            Card("G", 1),
            Card("Y", 1),
        ]
        self.cards = [*self.cards, *[c for c in build_cards_deck() if c not in self.cards]]
        self.cards = self.cards[::-1]
