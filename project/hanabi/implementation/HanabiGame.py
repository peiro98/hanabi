from typing import Optional, List, Tuple
import itertools
from functools import reduce
from operator import add

from Hint import ColorHint, Hint, ValueHint

from Card import Card, Deck
from Move import DiscardMove, HintColorMove, HintMove, PlayMove
from Player import *

HAND_SIZE = 5


class HanabiGame:
    """An Hanabi game."""

    def __init__(self) -> None:
        # list of players for this game
        self.player_proxies = []

        # played cards
        self.board = []
        # player hands
        self.hands = []
        # discard pile
        self.discard_pile = []

        # number of blue tokens left
        self.blue_tokens = 3
        # number of red tokens left
        self.red_tokens = 3

        # cards deck
        self.deck = Deck()

        self.game_started = False

    def register_player(self, player: "Player"):
        if self.game_started:
            raise ValueError("Game already started: can not add a new player!")

        # check whether player is already in the players list
        if player.name in (p.get_player().name for p in self.player_proxies):
            # if so, raise an error
            raise ValueError("Player is already registered!")

        self.player_proxies.append(PlayerGameProxy(player, self))

    def get_hand_of_player(self, player) -> List[Tuple[Card, List[Hint]]]:
        if player not in [proxy.get_player() for proxy in self.player_proxies]:
            raise ValueError("Invalid player")

        # TODO: please change the following line
        return self.hands[
            [proxy.get_player() for proxy in self.player_proxies].index(player)
        ]

    def get_board(self):
        return self.board

    def get_discard_pile(self):
        return self.discard_pile

    def __generate_hands(self):
        """Generate the initial hand of each player"""
        self.hands = [
            # (card, hints list)
            [(self.deck.pick_card(), set()) for _ in range(HAND_SIZE)]
            for _ in self.player_proxies
        ]

    def __is_playable(self, card: Card):
        """Check whether or not a card is compatible with the current state of the board"""
        # count the number of cards on the board with the same color
        n = len([c for c in self.board if c.color == card.color])

        # no card with the same color is already on the board
        # or card.value - 1 cards are already on the board
        return n == (card.value - 1)

    def __find_player_by_name(self, name: str):
        return next(
            (
                p.get_player()
                for p in self.player_proxies
                if p.get_player().name == name
            ),
            None,
        )

    def __apply_hint(self, player: Player, hint: Hint):
        idx = next(
            i for i, p in enumerate(self.player_proxies) if p.get_player() == player
        )

        self.hands[idx] = [
            (card, {reduce(add, hints | set([hint]))})
            if (hint.color == card.color or hint.value == card.value)
            else (card, hints)
            for card, hints in self.hands[idx]
        ]

    def score(self) -> int:
        return len(self.board)

    def start(self):
        self.__generate_hands()

        # iterate over players and hands until the game is over
        for proxy, hand in itertools.cycle(zip(self.player_proxies, self.hands)):
            if self.red_tokens < 0 or self.deck.is_empty():
                break

            # ask the player to perform a move
            move = proxy.step()

            # TODO: recompute the maximum achievable score after the move (here or in the proxy?)

            # TODO: move the move management to separate functions (or directly inside move)
            if isinstance(move, PlayMove):
                # extract the card to play
                card, _ = hand.pop(move.index)

                # TODO: replace with logging function
                print(f"Player {proxy.get_player()} plays {card}")

                if self.__is_playable(card):
                    # play the card and grant one blue token
                    self.board.append(card)
                else:
                    # move the card in the discard pile and remove one red token
                    self.discard_pile.append(card)
                    self.red_tokens -= 1

                # refill the hand
                hand.insert(0, (self.deck.pick_card(), set()))
            elif isinstance(move, DiscardMove):
                # extract the card to discard
                card, _ = hand.pop(move.index)

                print(f"Player {proxy.get_player()} discards {card}")

                # discard the card and grant one blue token
                self.discard_pile.append(card)
                self.blue_tokens += 1

                # refill the hand
                hand.insert(0, (self.deck.pick_card(), set()))
            elif isinstance(move, HintMove):
                # TODO: verify the player actually exist
                other = self.__find_player_by_name(move.player)

                print(f"Player {proxy.get_player()} hints {move}")

                if isinstance(move, HintColorMove):
                    self.__apply_hint(other, ColorHint(move.color))
                else:
                    self.__apply_hint(other, ValueHint(move.value))

                self.blue_tokens -= 1

            self.__print_state()

        print("Final score: ", self.score())

    def __print_state(self):
        print(f"Blue tokens: {self.blue_tokens:2d}.")
        print(f"Red tokens: {self.red_tokens:2d}")
        print(f"Board: {self.board}")
        print(f"Discard pile: {self.discard_pile}")

        print("Players:")
        for player_proxy, hand in zip(self.player_proxies, self.hands):
            player = player_proxy.get_player()
            print(f"- {player}")

            for card, hints in hand:
                print(f"  has {card} seen as {card.mask(hints)}")
        print()


class PlayerGameProxy:
    def __init__(self, player: Player, game: HanabiGame) -> None:
        self.player = player
        self.game = game

    def get_player(self) -> Player:
        return self.player

    def count_blue_tokens(self) -> int:
        return self.game.blue_tokens

    def get_other_players(self) -> List[Player]:
        # TODO: use a set
        return [
            p.get_player()
            for p in self.game.player_proxies
            if p.get_player() != self.player
        ]

    def step(self):
        return self.player.step(self)

    def see_hand(self, player: Optional[Player]) -> List[Tuple[Card, List[Hint]]]:
        hand = self.game.get_hand_of_player(player or self.player)

        if player != self.player:
            return hand

        return [(card.mask(hints), hints) for (card, hints) in hand]

    def see_board(self):
        return self.game.get_board()

    def see_discard_pile(self):
        return self.game.get_discard_pile()


if __name__ == "__main__":
    game = HanabiGame()
    game.register_player(PrompterAgent("Martha"))
    game.register_player(NaiveAgent("Jonas"))
    game.register_player(RiskAverseAgent("Ulrich"))
    game.register_player(NaiveAgent("Claudia"))
    game.register_player(ConstantAgent("Noah"))

    game.start()
