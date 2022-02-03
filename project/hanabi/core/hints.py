class Hint:
    """An hint

    An hint is a suggestion that can be shared among players.
    """

    def __init__(self, color: str, value: int, negative: bool = False) -> None:
        """Instantiate a new hint

        Parameters
        ----------
        color : str
            hint's color (one of "R", "Y", "G", "B", "W")
        value : int
            hint's value (one of 1, 2, 3, 4, 5)
        negative : bool
            whether the hint is negative or not
        """
        if color is not None and color not in ["R", "Y", "G", "B", "W"]:
            raise ValueError("Invalid value for the color parameter")
        if value is not None and value not in [1, 2, 3, 4, 5]:
            raise ValueError("Invalid value for the value parameter")
        if color is None and value is None:
            raise ValueError("At least one of the parameters color and value must be specified")

        self.color = color
        self.value = value
        self.negative = negative

    def __hash__(self) -> int:
        return hash(self.color) ^ hash(self.value)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, Hint)
            and self.color == __o.color
            and self.value == __o.value
            and self.negative == __o.negative
        )

    def __invert__(self):
        return Hint(self.color, self.value, not self.negative)

    def __str__(self) -> str:
        prefix = "~" if self.negative else ""

        if self.color:
            return f"{prefix}{self.color}"
        elif self.value:
            return f"{prefix}{self.value}"
        return f"{prefix}{self.color}-{self.value}"

    def __repr__(self) -> str:
        return self.__str__()


class ColorHint(Hint):
    """Color hint"""

    def __init__(self, color: str):
        super(__class__, self).__init__(color, None)


class ValueHint(Hint):
    """Value hint"""

    def __init__(self, value: int):
        super(__class__, self).__init__(None, value)
