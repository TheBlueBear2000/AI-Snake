from enum import Enum


class Directions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


print(Directions.UP)
print(Directions(Directions.UP.value + 1))
