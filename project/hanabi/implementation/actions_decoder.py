import random
from typing import Literal, Optional, Tuple

import numpy as np
from Card import CARD_COLORS, CARD_VALUES
from Move import Move, PlayMove, DiscardMove, HintColorMove, HintValueMove


class ActionDecoder:
    """Decode an action from a Q array

    The Q array stores the expected reward for all the available actions.
    The array is structured according to the following scheme:
     - indices `[0...4]` represent play actions (play(1), ..., play(5))
     - indices `[5...(len(Q) - 5 - 1)]` encode hint actions.
       For each player, ten actions are available: 5 for the color hints and
       5 for the value hints. Hint actions are encoded one player after the
       other, color hints first. Refer to `ActionDecoder.get` for more insights.
     - indices `[(len(Q) - 1):len(Q)]` encode discard actions (discard(1), ..., discard(5))
    """

    def __init__(self, Q: np.ndarray) -> None:
        """
        Initialize the action decoder.

        Parameters:
        -----------
        Q: np.ndarray
            output of the Q-learning model
        """
        self.Q = Q

    def size(self) -> int:
        """Return the size of the Q array

        Returns:
        int
            size of the Q array
        """
        return self.Q.size

    def pick_action(self, mode: Literal["max", "prob"] = "max") -> Tuple[Move, int]:
        """Pick an action

        The `mode` parameter specifies how the decoder should select an action
        to return. `max` returns the action with the highest corresponding Q value,
        `prob` returns a sample over the actions space using the probability
        distribution given by softmax(Q). `max` is the preferred mode.

        Parameters:
        -----------
        mode: Literal["max", "prob"]
            selection mode for the action (`prob` or `max`)

        Returns:
        --------
        Move
            selected action
        int
            selected action's index
        """

        if mode not in ["prob", "max"]:
            raise ValueError("Valid values for the mode parameter: ['max', 'prob']")

        if mode == "max":
            action_idx = np.argmax(self.Q, 0)[0]
        else:
            # transform Q values into probabilities
            probs = np.copy(self.Q)
            # do not consider values below 1e-5
            probs[probs < 1e-5] = 0.0
            # apply softmax function to Q (convert Q values to probabilities)
            probs[probs > 0] = np.exp(probs[probs > 0]) / np.sum(np.exp(probs[probs > 0]))
            # pick an action from the categorical distribution defined by probs
            action_idx = np.random.choice(range(len(self.Q)), p=probs)

        # return the action and the corresponding index in the Q array
        return self.get(action_idx), action_idx

    def pick_random(self, probabilities: Optional[np.ndarray] = np.array([1 / 3, 1 / 3, 1 / 3])) -> Tuple[Move, int]:
        """Pick a random action given the current Q array

        Parameters:
        -----------
        probabilities: Optional[np.ndarray]
            probabilities of selection an action of certain type: ["play", "hint", "discard]

        Returns:
        --------
        Move
            selected action
        int
            selected action's index
        """

        # "move", "hint" and "discard" have initially the same probability
        c = np.random.choice([0, 1, 2], p=probabilities)
        if c == 0:
            # random move
            action_idx = random.randint(0, 4)
        elif c == 1:
            # random hint
            action_idx = random.randint(5, self.size() - 5 - 1)
        else:
            # random discard
            action_idx = random.randint(self.size() - 5, self.size() - 1)

        return self.get(action_idx), action_idx

    def get(self, idx: int) -> Move:
        """Convert an action index to a proper move object

        Parameters:
        -----------
        idx: int
            index of the action to be decoded

        Returns:
        --------
        Move
            decoded action
        """
        n_players = (self.size() - 10) // 10

        if idx < 5:
            # first five values => play action
            return PlayMove(idx)

        idx = idx - 5
        if idx < 5 * n_players:
            # next 5 * (n_players - 1) values => hint value action
            return HintValueMove(idx // 5, CARD_VALUES[idx % 5])

        idx = idx - 5 * n_players
        if idx < 5 * n_players:
            # next 5 * (n_players - 1) values => hint color action
            return HintColorMove(idx // 5, CARD_COLORS[idx % 5])

        idx = idx - 5 * n_players
        # last five values => discard action
        return DiscardMove(idx)
