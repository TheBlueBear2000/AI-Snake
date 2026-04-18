DIRECTIONS = ["up", "right", "down", "left"]
changes = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]

for direction in DIRECTIONS:
    print("when going", direction)
    print(
        list(range(DIRECTIONS.index(direction) * 2, len(changes)))
        + list(range(0, DIRECTIONS.index(direction) * 2))
    )
