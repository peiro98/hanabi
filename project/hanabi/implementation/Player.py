from __future__ import annotations
from distutils.command.config import config

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import sys
import time
import logging

from Hint import ColorHint
from Move import Move, PlayMove, DiscardMove, HintMove, HintColorMove, HintValueMove
from Card import CARD_COLORS, Card

from state_encoder import FlatStateEncoder, StateEncoder
from actions_decoder import ActionDecoder

import numpy as np

# See https://stackoverflow.com/a/39757388
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HanabiGame import PlayerGameProxy


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
            [p for p in proxy.get_other_players() if any(len(hints) < 2 for _, hints in proxy.see_hand(p))]
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
            (
                i
                for i, (card, _) in enumerate(my_hand)
                if (card.color and card.value) or (card.value == 1 and all(c.value == 1 for c in playable_cards))
            ),
            None,
        )
        if card_index is not None:
            return PlayMove(card_index)

        if proxy.count_blue_tokens():
            return self.hint_playable(proxy) or self.give_random_hint(proxy)
            # return self.give_random_hint(proxy)

        return DiscardMove(random.randint(0, len(my_hand) - 1))


class DRLNonTrainableAgent(Player):
    """Non-trainable Deep Reinforcement Learning player"""

    def __init__(
        self,
        name: str,
        *,
        filenames: Dict[int, str] = None,
        models: Dict[int, List[np.ndarray]] = None,
        state_encoder_builder: Any = FlatStateEncoder,
    ) -> None:
        """Instantiate the player

        Parameters
        ----------
        name : str
            name of the player
        filenames : Dict[int, str]
            dictionary containing the filenames of the models parameters.
            This parameter is mutually exclusive with the models parameter
        models : Dict[int, List[np.ndarray]]
            models parameters, one list of numpy arrays for each possible number
            of players. This parameter is mutually exclusive with the
            models parameter
        state_encoder_builder : Any, optional
            an instance of StateEncoder that encodes the board state into a
            numpy array that can be fed to a nn, by default FlatStateEncoder
        """
        super().__init__(name)

        if (filenames and models) or (not filenames and not models):
            raise ValueError("Exactly one of [filenames, models] must be not None")

        if models:
            self.models = models
        elif filenames:
            self.models = {n: self.load_model(filename) for n, filename in filenames.items()}

        self.state_encoder_builder = state_encoder_builder

    @staticmethod
    def load_model(filename: str) -> List[np.ndarray]:
        """Load model parameters from a numpy file

        Weights and biases are stored in the file two by two:
           W', b', W'', b''
        where W' and b' represents the weights matrix and bias
        vector of the first linear layers of the network.

        Parameters
        ----------
        filename : str
            path to the model parameter file

        Returns
        -------
        List[np.ndarray]
            [description]
        """
        parameters = []
        with open(filename, "rb") as f:
            while True:
                try:
                    parameters.append(np.load(f))
                except:
                    break

        return parameters

    def encode_current_state(self, proxy: "PlayerGameProxy") -> StateEncoder:
        """Encode the current game state"""
        players_state = [proxy.see_hand(p) for p in [proxy.get_player(), *proxy.get_other_players()]]
        board_state = proxy.see_board()
        discard_pile_state = proxy.see_discard_pile()

        return self.state_encoder_builder(
            players_state,
            board_state,
            discard_pile_state,
            proxy.count_blue_tokens(),
            proxy.count_red_tokens(),
        )

    def compute_Q(self, state_encoder: StateEncoder) -> np.ndarray:
        """Compute the Q output

        The underlying model is an MLP (Multilayer Perceptron)
        whose linear layers are followed by a ReLu operation.
        ReLu is not applied to the last layer.

        Parameters
        ----------
        state_encoder : StateEncoder
            state encoder for the board

        Returns
        -------
        np.ndarray
            Q values

        Raises
        ------
        ValueError
            if the number of players playing current game is not supported
            by any of its models
        """
        n = state_encoder.n_players()

        if n not in self.models:
            logging.critical(f"{self.name} does not support {n} players")
            raise ValueError("Invalid number of players")

        model = self.models[n]

        Ws, bs = model[0::2], model[1::2]
        x = state_encoder.get()

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        for i, (W, b) in enumerate(zip(Ws, bs)):
            x = x @ W.T + b
            if i < (len(Ws) - 1):
                x[x < 0] = 0

        return x.flatten()

    def is_valid_move(self, action: Move, proxy: "PlayerGameProxy") -> bool:
        """Verify whether an action is valid or not.

        Parameters
        ----------
        action : Move
            the action to test
        proxy : PlayerGameProxy
            a proxy to the runnning game

        Returns
        -------
        bool
            validity of the move
        """
        if action is None:
            return False

        if isinstance(action, HintMove) and proxy.count_blue_tokens() <= 0:
            # not enough blue tokens
            return False

        if isinstance(action, PlayMove) and len(proxy.see_hand()) <= action.index:
            # invalid move's index
            return False

        if isinstance(action, DiscardMove) and len(proxy.see_hand()) <= action.index:
            # invalid move's index
            return False

        # TODO: are more checks required?
        return True

    def step(self, proxy: "PlayerGameProxy"):
        # Extract the current state from the game's proxy
        encoded_state = self.encode_current_state(proxy)

        Q = self.compute_Q(encoded_state)
        # Instantiate an action decoder for the network output
        decoder = ActionDecoder(Q)

        action = None
        while not self.is_valid_move(action, proxy):
            if action is not None:
                logging.debug(f"{self.name} selected an invalid action {action}")
                # the network suggested to perform an invalid action => panic
                action, _ = decoder.pick_random()
            else:
                action, _ = decoder.pick_action(mode="max")

            # hint actions refer to players using their indices
            if isinstance(action, HintMove):
                # convert indices to names
                action.player = proxy.get_other_players()[action.player].name

        logging.debug(f"{self.name} selected action {action}")

        return action


if __name__ == "__main__":
    filename = time.strftime("%Y_%m_%d-%I_%M_%S_%p")
    logging.basicConfig(
        filename=f"logs/{filename}.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Run a game
    from HanabiGame import HanabiGame

    game = HanabiGame()

    players = [
        DRLNonTrainableAgent(f"P{i}", {2: "project/hanabi/implementation/models/drl_2_players.npy"})
        for i in range(2)
    ]

    game.register_player(players[0])
    game.register_player(players[1])

    game.start()

    logging.info(f"Game result: {game.score()}")
