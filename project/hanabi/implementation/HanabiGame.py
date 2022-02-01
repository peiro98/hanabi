import logging
from typing import Optional, List, Tuple
import itertools

from Hint import ColorHint, Hint, ValueHint
from Card import Card, Deck
from Move import DiscardMove, HintColorMove, HintMove, PlayMove
from Player import *

HAND_SIZE = 5

REWARDS = {
    "PLAYED_WRONG_CARD": -1,
    "PLAYED_CARD_WITHOUT_HINTS": -0.75,
    "PLAYED_CARD_WITH_ONLY_ONE_HINT": -0.25,
    "DISCARDED_CARD_WITHOUT_HINTS": -0.5,
    "DISCARDED_CARD_WITH_ONLY_ONE_HINT": -0.2,
    "DISCARDED_UNPLAYABLE_CARD": +0.05,
    "DISCARDED_PLAYABLE_CARD": -0.5,
    "PLAYED_CORRECT_CARD": +1,
    "HINTED_CARD_WITHOUT_PREVIOUS_HINTS": +0.2,
    "HINTED_CARD_WITH_ONE_PREVIOUS_HINT": +0.1,
    "HINTED_FULLY_KNOWN_CARD": -0.75,
    "ILLEGAL": -1
    # "PLAYED_WRONG_CARD": 0,
    # "PLAYED_CARD_WITHOUT_HINTS": 0,
    # "PLAYED_CARD_WITH_ONLY_ONE_HINT": 0,
    # "DISCARDED_CARD_WITHOUT_HINTS": 0,
    # "DISCARDED_CARD_WITH_ONLY_ONE_HINT": 0,
    # "DISCARDED_UNPLAYABLE_CARD": 0,
    # "DISCARDED_PLAYABLE_CARD": 0,
    # "PLAYED_CORRECT_CARD": 2,
    # "HINTED_CARD_WITHOUT_PREVIOUS_HINTS": 0,
    # "HINTED_CARD_WITH_ONE_PREVIOUS_HINT": 0,
    # "HINTED_FULLY_KNOWN_CARD": 0,
    # "ILLEGAL": 0
}


class HanabiGame:
    """An Hanabi game."""

    def __init__(self, deck=None, *, verbose=True) -> None:
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
        self.deck = deck or Deck()

        self.game_started = False
        self.iter_index = 0
        self.verbose = verbose

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
        return self.hands[[proxy.get_player() for proxy in self.player_proxies].index(player)]

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
            (p.get_player() for p in self.player_proxies if p.get_player().name == name),
            None,
        )

    def __apply_hint(self, player: Player, hint: Hint):
        idx = next(i for i, p in enumerate(self.player_proxies) if p.get_player() == player)

        self.hands[idx] = [
            (card, hints | set([hint])) if (hint.color == card.color or hint.value == card.value) else (card, hints)
            for card, hints in self.hands[idx]
        ]

    def score(self) -> int:
        return len(self.board)

    def start(self, *, early_stop_at=None):
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
                card, hints = hand.pop(move.index)

                if self.__is_playable(card):
                    # play the card and grant one blue token
                    self.board.append(card)
                    for p in self.player_proxies:
                        if p.get_player() == proxy.get_player():
                            reward = REWARDS["PLAYED_CORRECT_CARD"]
                            logging.debug(f"Correct card played: assigning {reward} to {p.get_player().name}")
                            p.reward_player(reward)
                        else:
                            reward = REWARDS["PLAYED_CORRECT_CARD"] * 0.5
                            logging.debug(f"Correct card played: assigning {reward} to {p.get_player().name}")
                            p.reward_player(REWARDS["PLAYED_CORRECT_CARD"])

                    if card.value == 5:
                        self.blue_tokens += 1
                else:
                    # move the card in the discard pile and remove one red token
                    self.discard_pile.append(card)
                    self.red_tokens -= 1
                    reward = REWARDS["PLAYED_WRONG_CARD"]
                    logging.debug(f"Wrong card played: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(reward)

                if len(hints) == 0:
                    reward = REWARDS["PLAYED_CARD_WITHOUT_HINTS"]
                    logging.debug(f"Played card without hints: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(reward)
                elif len(hints) == 1:
                    reward = REWARDS["PLAYED_CARD_WITH_ONLY_ONE_HINT"]
                    logging.debug(f"Played card with only one hint: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(reward)

                # refill the hand
                hand.insert(0, (self.deck.pick_card(), set()))
            elif isinstance(move, DiscardMove):
                # extract the card to discard
                card, hints = hand.pop(move.index)

                # discard the card and grant one blue token
                self.discard_pile.append(card)
                self.blue_tokens += 1

                if len(hints) == 0:
                    reward = REWARDS["DISCARDED_CARD_WITHOUT_HINTS"]
                    logging.debug(f"Discarded card without hints: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(REWARDS["DISCARDED_CARD_WITHOUT_HINTS"])
                elif len(hints) == 1:
                    reward = REWARDS["DISCARDED_CARD_WITH_ONLY_ONE_HINT"]
                    logging.debug(
                        f"Discarded card with only one hints: assigning {reward} to {proxy.get_player().name}"
                    )
                    proxy.reward_player(REWARDS["DISCARDED_CARD_WITH_ONLY_ONE_HINT"])

                if not self.__is_playable(card):
                    reward = REWARDS["DISCARDED_UNPLAYABLE_CARD"]
                    logging.debug(f"Discarded unplayable card: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(REWARDS["DISCARDED_UNPLAYABLE_CARD"])
                else:
                    reward = REWARDS["DISCARDED_PLAYABLE_CARD"]
                    logging.debug(f"Discarded playable card: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(REWARDS["DISCARDED_PLAYABLE_CARD"])

                # refill the hand
                hand.insert(0, (self.deck.pick_card(), set()))
            elif isinstance(move, HintMove):
                # TODO: verify the player actually exist
                other = self.__find_player_by_name(move.player)

                if self.blue_tokens <= 0 or proxy.get_player() == other:
                    reward = REWARDS["ILLEGAL"]
                    logging.debug(f"Illegal hint: assigning {reward} to {proxy.get_player().name}")
                    proxy.reward_player(reward)
                    break

                if isinstance(move, HintColorMove):
                    self.__apply_hint(other, ColorHint(move.color))
                else:
                    self.__apply_hint(other, ValueHint(move.value))

                self.blue_tokens -= 1

            self.__print_state()

            self.iter_index += 1
            if early_stop_at and self.iter_index == (early_stop_at * len(self.player_proxies)):
                break

        for proxy in self.player_proxies:
            player = proxy.get_player()
            if callable(getattr(player, "train", None)):
                player.train(proxy)

    def get_turn_index(self):
        return self.iter_index // len(self.player_proxies)

    def __print_state(self):
        logging.debug(f"After {self.iter_index} steps the board state is")
        logging.debug(f"  Blue tokens: {self.blue_tokens:2d}")
        logging.debug(f"  Red tokens: {self.red_tokens:2d}")
        logging.debug(f"  Board: {self.board}")
        logging.debug(f"  Discard pile: {self.discard_pile}")

        logging.debug(f"  Players:")
        for player_proxy, hand in zip(self.player_proxies, self.hands):
            player = player_proxy.get_player()

            str_hand = ", ".join(str((card, hints if len(hints) > 0 else {})) for card, hints in hand)
            logging.debug(f"    {player}: [{str_hand}]")


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
        return [p.get_player() for p in self.game.player_proxies if p.get_player() != self.player]

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
        if isinstance(self.player, TrainablePlayer) and len(self.player.rewards) > 0:
            self.player.receive_reward(reward)

    def get_turn_index(self):
        return self.game.get_turn_index()
