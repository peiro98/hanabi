from abc import abstractmethod
from typing import List, Tuple
import random

import torch


class ReplayMemory:
    """Store past experience of a Deep Q-Learning agent

    In the Deep Q-Learning context, replay memory is used to preserve
    the experience obtained by an agent so it can be replayed again in
    the future. Replay memory improves generalization of the models.
    """

    def __init__(self) -> None:
        """Instantiate the replay memory"""
        self.experiences = []

    def add(self, state: torch.tensor, action: int, reward: float, next_state: torch.tensor):
        """Add a new tuple (state, reward, action, next_state) to the memory

        Parameters
        ----------
        state : torch.tensor
            state seen by the agent
        action : int
            index of the action taken by the agent
        reward : float
            reward obtained by the agent
        next_state : torch.tensor
            next state observed by the agent
        """
        self.experiences.append((state, action, reward, next_state))
        # Keep the size of the experience under control
        self.trim_experience()

    def add_experience_from_game(self, states: List[torch.tensor], actions: List[int], rewards: List[float]):
        """Add the experience collected during a game to the replay memory

        Parameters
        ----------
        states : List[torch.tensor]
            list of states observed by the agent
        actions : List[int]
            indices of the selected action
        rewards : List[float]
            rewards obtained by the agent
        """
        for s, a, r, ns in zip(states, actions, rewards, states[1:] + [None]):
            self.add(s, a, r, ns)

    def size(self) -> int:
        """Return the current size of the replay memory

        Returns
        -------
        int
            current size of the replay memory
        """
        return len(self.experiences)

    def __len__(self) -> int:
        """Return the current size of the replay memory

        Returns
        -------
        int
            current size of the replay memory
        """
        return self.size()

    def trim_experience(self):
        """Keep the size of the replay memory under control"""
        pass

    @abstractmethod
    def sample(self, size):
        raise NotImplementedError()


class UniformReplayMemory(ReplayMemory):
    """Replay memory that does not bias experience depending on the reward"""

    def __init__(self, max_size: int, *args, **kwargs) -> None:
        """Instantiate the replay memory

        Parameters
        ----------
        max_size : int
            maximum size of the replay memory
        """
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def trim_experience(self):
        """Remove old experiences"""
        self.experiences = self.experiences[-self.max_size:]

    def sample(self, size: int) -> List[Tuple[torch.tensor, int, float, torch.tensor]]:
        """Sample a subset of the past experiences

        Parameters
        ----------
        size : int
            batch size

        Returns
        -------
        List[Tuple[torch.tensor, float, int, torch.tensor]]
            sampled past experiences
        """
        size = min(size, len(self.experiences))
        return random.sample(self.experiences, size)
