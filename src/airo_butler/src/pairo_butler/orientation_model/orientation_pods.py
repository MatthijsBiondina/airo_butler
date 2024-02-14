from pathlib import Path
from typing import Union


class Coord:
    __slots__ = ["x", "y"]

    def __init__(self, x: Union[int, float], y: Union[int, float]):
        assert type(x) == type(y)

        self.x: Union[int, float] = x
        self.y: Union[int, float] = y


class SampleMetaDataPOD:
    __slots__ = ["path", "center", "angle"]

    def __init__(self, path: Path, center: Coord, angle: float):
        self.path: Path = path
        self.center: Coord = center
        self.angle: float = angle
