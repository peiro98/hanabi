from ast import Constant
import statistics
from typing import Optional, List, Tuple
import itertools
from functools import reduce
from operator import add
from Hint import ColorHint, Hint, ValueHint
from random import shuffle

from Card import Card, Deck
from Move import DiscardMove, HintColorMove, HintMove, PlayMove
from Player import *
from Learning import DRLAgent

HAND_SIZE = 5

eps = 0.05

REWARDS = {
    # "PLAYED_WRONG_CARD": -10,
    # "PLAYED_CARD_WITHOUT_HINTS": -7.5,
    # "PLAYED_CARD_WITH_ONLY_ONE_HINT": -2.5,
    # "DISCARDED_CARD_WITHOUT_HINTS": -5,
    # "DISCARDED_CARD_WITH_ONLY_ONE_HINT": -2,
    # "DISCARDED_UNPLAYABLE_CARD": 0,
    # "DISCARDED_PLAYABLE_CARD": -0.5,
    # "PLAYED_CORRECT_CARD": 10,
    # "HINTED_CARD_WITHOUT_PREVIOUS_HINTS": 0.2,
    # "HINTED_CARD_WITH_ONE_PREVIOUS_HINT": 0.1,
    # "HINTED_FULLY_KNOWN_CARD": -5,
    # "ILLEGAL": -25
    "PLAYED_WRONG_CARD": 0,
    "PLAYED_CARD_WITHOUT_HINTS": 0,
    "PLAYED_CARD_WITH_ONLY_ONE_HINT": 0,
    "DISCARDED_CARD_WITHOUT_HINTS": 0,
    "DISCARDED_CARD_WITH_ONLY_ONE_HINT": 0,
    "DISCARDED_UNPLAYABLE_CARD": 0,
    "DISCARDED_PLAYABLE_CARD": 0,
    "PLAYED_CORRECT_CARD": 1,
    "HINTED_CARD_WITHOUT_PREVIOUS_HINTS": 0,
    "HINTED_CARD_WITH_ONE_PREVIOUS_HINT": 0,
    "HINTED_FULLY_KNOWN_CARD": 0,
    "ILLEGAL": 0
}


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
        self.blue_tokens = 8
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

        illegal = False

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
                card, hints = hand.pop(move.index)

                # TODO: replace with logging function
                print(f"Player {proxy.get_player()} plays {card}")

                if self.__is_playable(card):
                    # play the card and grant one blue token
                    self.board.append(card)
                    proxy.reward_player(REWARDS["PLAYED_CORRECT_CARD"])
                else:
                    # move the card in the discard pile and remove one red token
                    self.discard_pile.append(card)
                    self.red_tokens -= 1
                    proxy.reward_player(REWARDS["PLAYED_WRONG_CARD"])

                if len(hints) == 0:
                    proxy.reward_player(REWARDS["PLAYED_CARD_WITHOUT_HINTS"])
                elif len(hints) == 1:
                    proxy.reward_player(REWARDS["PLAYED_CARD_WITH_ONLY_ONE_HINT"])

                # refill the hand
                hand.insert(0, (self.deck.pick_card(), set()))
            elif isinstance(move, DiscardMove):
                # extract the card to discard
                card, hints = hand.pop(move.index)

                print(f"Player {proxy.get_player()} discards {card}")

                # discard the card and grant one blue token
                self.discard_pile.append(card)
                self.blue_tokens += 1

                if len(hints) == 0:
                    proxy.reward_player(REWARDS["DISCARDED_CARD_WITHOUT_HINTS"])
                elif len(hints) == 1:
                    proxy.reward_player(REWARDS["DISCARDED_CARD_WITH_ONLY_ONE_HINT"])

                if not self.__is_playable(card):
                    proxy.reward_player(REWARDS["DISCARDED_UNPLAYABLE_CARD"])
                else:
                    proxy.reward_player(REWARDS["DISCARDED_PLAYABLE_CARD"])

                # refill the hand
                hand.insert(0, (self.deck.pick_card(), set()))
            elif isinstance(move, HintMove):
                # TODO: verify the player actually exist
                other = self.__find_player_by_name(move.player)

                if self.blue_tokens <= 0 or proxy.get_player() == other:
                    proxy.reward_player(REWARDS["ILLEGAL"])
                    # illegal = True
                    # break

                print(f"Player {proxy.get_player()} hints {move}")

                if isinstance(move, HintColorMove):
                    self.__apply_hint(other, ColorHint(move.color))
                else:
                    self.__apply_hint(other, ValueHint(move.value))

                self.blue_tokens -= 1

            # self.__print_state()

        for p in self.player_proxies:
            # p.reward_player(self.score())
            if isinstance(p.get_player(), DRLAgent):
                p.get_player().train()
        
        print("Final score: ", self.score() )

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

    def count_red_tokens(self) -> int:
        return self.game.red_tokens

    def get_other_players(self) -> List[Player]:
        # TODO: use a set
        return [
            p.get_player()
            for p in self.game.player_proxies
            if p.get_player() != self.player
        ]

    def step(self):
        global eps
        return self.player.step(self)

    def give_reward(self, reward: float):
        self.player.receive_reward(reward)

    def see_hand(self, player: Optional[Player]) -> List[Tuple[Card, List[Hint]]]:
        hand = self.game.get_hand_of_player(player or self.player)

        if player != self.player:
            return hand

        return [(card.mask(hints), hints) for (card, hints) in hand]

    def see_board(self):
        return self.game.get_board()

    def see_discard_pile(self):
        return self.game.get_discard_pile()

    def reward_player(self, reward):
        if isinstance(self.player, TrainablePlayer):
            self.player.receive_reward(reward)


if __name__ == "__main__":
    players = [
        DRLAgent("Martha", n_players=2, training=True),
        #DiscardAgent("Jonas")
        DRLAgent("Jonas", n_players=2, training=True),
        #DRLAgent("Ulrich", n_players=4, training=True),
        #DRLAgent("Claudia", n_players=4, training=True),
        # DRLAgent("Noah", n_players=4)
    ]

    scores = []
    for i in range(100000):
        game = HanabiGame()
        print(f"#{i}")

        shuffle(players)
        for p in players:
            #p.training = (i // 50) % 2 == 0
            if isinstance(p, DRLAgent):
                p.prepare()
            game.register_player(p)
        game.start()

        #eps = eps * 0.9995
        # print(eps)

        scores.append(game.score())
        print(f"Mean: {statistics.mean(scores)}")
