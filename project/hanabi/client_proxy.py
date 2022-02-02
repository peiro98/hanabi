import logging
from typing import List, Optional, Tuple, Union

from implementation.cards import Card
from implementation.hints import Hint, ValueHint, ColorHint
from implementation.hanabi_game import PlayerGameProxy
from implementation.players import Player, UnknownPlayer
from implementation.moves import Move


class Proxy(PlayerGameProxy):
    """Glue together an agent and a game environment

    A proxy is required since agents are defined in a game environment
    independent way. A proxy allows the agent to access relevant
    informations it requires in order to take its decisions.
    """

    def __init__(self, player: Player) -> None:
        """Instantiate a proxy

        Parameters
        ----------
        player : Player
            the players that access the game env through this proxy
        """
        self.player = player
        self.clean()

    def get_player(self) -> Player:
        return self.player

    def clean(self):
        """Prepare the proxy for a new game"""
        self.players = {
            # player_name: (player, hand)
            self.player.name: (self.player, [])
        }
        self.board = []
        self.discard_pile = []

        # number of blue tokens left
        self.blue_tokens = 8
        # number of red tokens left
        self.red_tokens = 3

    def add_player(self, playerName: str):
        """Add a new external *uncontrolled* player to the proxy

        Parameters
        ----------
        playerName : str
            player's name
        """
        self.players[playerName] = (UnknownPlayer(playerName), [])

    def count_blue_tokens(self) -> int:
        """Return the number of blue tokens available

        Returns
        -------
        int
            the number of available blue tokens
        """
        return self.blue_tokens

    def count_red_tokens(self) -> int:
        """Return the number of red tokens available

        Returns
        -------
        int
            the number of available red tokens
        """
        return self.red_tokens

    def get_other_players(self) -> List[Player]:
        """Return the list of uncontrolled players

        Returns
        -------
        List[Player]
            a list of uncontrolled players participating in the same game
        """
        return [player for _, (player, hand) in self.players.items() if player != self.player]

    def step(self) -> Move:
        """Ask the player to take an action

        Returns
        -------
        Move
            a valid move given the current game state
        """
        # pass the proxy to the player's step method so that it
        # can access the game state through this object
        return self.player.step(self)

    def see_hand(self, player: Optional[Player] = None) -> List[Tuple[Card, List[Hint]]]:
        """See the hand of a player

        Parameters
        ----------
        player : Optional[Player], optional
            player whose hand is to be seen, by default None (self)

        Returns
        -------
        List[Tuple[Card, List[Hint]]]
            the hand as a list of tuples (card, [hint])
        """
        player = player or self.player
        _, hand = self.players[player.name]

        if player != self.player:
            return hand

        # Cards are masked according to the available hints (avoid
        # leaking to the player information it shouldn't be able to
        # access). This is important in simulated environments in particular.
        # In such scenario, the hand variable at this point may contain
        # fully known card, regardless of the hints. Refer to the implementation
        # of HanabiGame in implementation/hanabi_game.py for more details.
        return [(card.mask(hints), hints) for (card, hints) in hand]

    def see_board(self) -> List[Card]:
        """Return the list of cards on the board

        Returns
        -------
        List[Card]
            list of played cards
        """
        return self.board

    def see_discard_pile(self) -> List[Card]:
        """Return the list of discarded cards

        Returns
        -------
        List[Card]
            list of discarded cards
        """
        return self.discard_pile

    def append_card_to_board(self, color: str, value: int):
        """Add a new card to the board

        Add a new card to the board (checks are not performed here).

        Parameters
        ----------
        color : str
            color of the card to add
        value : int
            value of the card to add
        """
        self.board.append(Card(color, value))

    def append_card_to_discard_pile(self, color: str, value: int):
        """Add a new card to the discard pile

        Add a new card to the discard pile (checks are not performed here).

        Parameters
        ----------
        color : str
            color of the card to add
        value : int
            value of the card to add
        """
        color = color[0].upper()  # make sure the color is a single uppercase char
        self.discard_pile.append(Card(color, value))

    def append_card_to_player_hand(self, playerName: str, color: str, value: str):
        """Add a new card to the player's hand

        Parameters
        ----------
        playerName : str
            player's name
        color : str
            color of the card to add
        value : str
            value of the card to add
        """
        _, hand = self.players[playerName]

        if color is not None:
            color = color[0].upper()  # make sure the color is a single uppercase char

        card = Card(color, value)
        hand.append((card, []))  # (card, [hint])

    def hint_player(self, playerName: str, type: str, value: Union[int, str], positions: List[int]):
        """Hint a player

        Hint `playerName` that all its cards in positions `positions`
        are have `type` ('color' or 'value') equal to `value`

        Parameters
        ----------
        playerName : str
            player's name
        type : str
            hint type ('color' or 'value')
        value : Union[int, str]
            value to hint
        positions : List[int]
            positions of the card satisfying the hint
        """
        if playerName not in self.players:
            raise ValueError("Unknown player")

        _, hand = self.players[playerName]

        logging.info(str(hand) + str(positions))

        for i in range(len(hand)):
            if type == "color":
                color = value[0].upper()  # make sure the color is a single uppercase char
                hint = ColorHint(color)
            else:
                hint = ValueHint(value)
            if i in positions and hint not in hand[i][1]:
                hand[i][1].append(hint)
            if i not in positions and hint not in hand[i][1]:
                hand[i][1].append(~hint)
