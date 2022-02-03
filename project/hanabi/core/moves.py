from typing import Union

class Move:
    pass

class PlayMove(Move):
    def __init__(self, index) -> None:
        # TODO: the constraints must depend on the player's hand size (dynamic)
        if index not in range(5):
            raise ValueError("Index not in [0, 4]")
        self.index = index

    def __str__(self) -> str:
        return f"PlayMove({self.index})"

    def __repr__(self) -> str:
        return self.__str__()


class DiscardMove(Move):
    def __init__(self, index) -> None:
        # TODO: the constraints must depend on the player's hand size (dynamic)
        if index not in range(5):
            raise ValueError("Index not in [0, 4]")
        self.index = index

    def __str__(self) -> str:
        return f"DiscardMove({self.index})"

    def __repr__(self) -> str:
        return self.__str__()


class HintMove(Move):
    def __init__(self, player: str) -> None:
        # TODO: dynamically verify the player
        self.player = player


class HintColorMove(HintMove):
    def __init__(self, player: str, color: str) -> None:
        # TODO: dynamically verify the color
        super(__class__, self).__init__(player)
        self.color = color

    def __str__(self) -> str:
        return f"HintColorMove({self.player}, {self.color})"

    def __repr__(self) -> str:
        return self.__str__()


class HintValueMove(HintMove):
    def __init__(self, player: Union[str, int], value: str) -> None:
        # TODO: dynamically verify the value
        super(__class__, self).__init__(player)
        self.value = value

    def __str__(self) -> str:
        return f"HintValueMove({self.player}, {self.value})"

    def __repr__(self) -> str:
        return self.__str__()
