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
        pass

    @abstractmethod
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
        raise NotImplementedError()

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

    @abstractmethod
    def size(self) -> int:
        """Return the current size of the replay memory

        Returns
        -------
        int
            current size of the replay memory
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the current size of the replay memory

        Returns
        -------
        int
            current size of the replay memory
        """
        return self.size()

    @abstractmethod
    def trim_experience(self):
        """Keep the size of the replay memory under control"""
        raise NotImplementedError()

    @abstractmethod
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
        self.experiences = []

    def add(self, state: torch.tensor, action: int, reward: float, next_state: torch.tensor):
        self.experiences.append((state, action, reward, next_state))
        # Keep the size of the experience under control
        self.trim_experience()

    def size(self) -> int:
        return len(self.experiences)

    def trim_experience(self):
        self.experiences = self.experiences[-self.max_size :]

    def sample(self, size: int) -> List[Tuple[torch.tensor, int, float, torch.tensor]]:
        size = min(size, len(self.experiences))
        return random.sample(self.experiences, size)


class PositiveNegativeReplayMemory(ReplayMemory):
    """Keep positive and negative experiences separate"""

    def __init__(self, max_size_positive: int, max_size_negative: int, sampling_positive_percentage: float) -> None:
        """Initialize PositiveNegativeReplayMemory

        Parameters
        ----------
        max_size_positive : int
            max size of the positive memory
        max_size_negative : int
            max size of the negative memory
        sampling_positive_percentage : float
            percentage of positive samples in returned by the .sample method
        """
        self.max_size_positive = max_size_positive
        self.max_size_negative = max_size_negative
        self.sampling_positive_percentage = sampling_positive_percentage

        # initialize the memories
        self.positive_experiences = []
        self.negative_experiences = []

    def add(self, state: torch.tensor, action: int, reward: float, next_state: torch.tensor):
        if reward > 0:
            self.positive_experiences.append((state, action, reward, next_state))
        else:
            self.negative_experiences.append((state, action, reward, next_state))
        self.trim_experience()

    def trim_experience(self):
        self.positive_experiences = self.positive_experiences[-self.max_size_positive :]
        self.negative_experiences = self.negative_experiences[-self.max_size_negative :]

    def size(self) -> int:
        return len(self.positive_experiences) + len(self.negative_experiences)

    def sample(self, size: int) -> List[Tuple[torch.tensor, int, float, torch.tensor]]:
        positive_size = min(len(self.positive_experiences), int(size * self.sampling_positive_percentage))
        negative_size = min(len(self.negative_experiences), size - positive_size)

        samples = random.sample(self.positive_experiences, positive_size)
        samples += random.sample(self.negative_experiences, negative_size)
        return samples
