class Hint:
    """An hint

    An hint is a suggestion that can be shared among players.
    """

    def __init__(self, color: str, value: int) -> None:
        self.color = color
        self.value = value

    def __hash__(self) -> int:
        return hash(self.color) ^ hash(self.value)

    def __add__(self, other):
        if not isinstance(other, Hint):
            raise ValueError("Only hints can be summed")

        return Hint(self.color or other.color, self.value or other.value)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, Hint)
            and self.color == __o.color
            and self.value == __o.value
        )

    def __str__(self) -> str:
        if self.color:
            return f"{self.color}"
        elif self.value:
            return f"{self.value}"
        return f"{self.color}-{self.value}"

    def __repr__(self) -> str:
        return self.__str__()


class ColorHint(Hint):
    def __init__(self, color: str):
        super(__class__, self).__init__(color, None)

    def __hash__(self) -> int:
        return super().__hash__()


class ValueHint(Hint):
    def __init__(self, value: int):
        super(__class__, self).__init__(None, value)

    def __hash__(self) -> int:
        return super().__hash__()

