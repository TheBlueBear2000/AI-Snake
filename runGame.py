from random import randint

ARENA_DIMS = (20, 15)


def doMove(snake, apples, direction):
    if direction == None:
        return {"snake": snake, "apples": apples, "alive": True, "won": False}

    if len(snake) == ARENA_DIMS[0] * ARENA_DIMS[1]:
        return {"snake": snake, "apples": apples, "alive": True, "won": True}

    if direction == "up":
        new_coordinate = (snake[-1][0], snake[-1][1] - 1)
    elif direction == "down":
        new_coordinate = (snake[-1][0], snake[-1][1] + 1)
    elif direction == "left":
        new_coordinate = (snake[-1][0] - 1, snake[-1][1])
    elif direction == "right":
        new_coordinate = (snake[-1][0] + 1, snake[-1][1])

    if (
        new_coordinate in snake
        or new_coordinate[0] < 0
        or new_coordinate[0] >= ARENA_DIMS[0]
        or new_coordinate[1] < 0
        or new_coordinate[1] >= ARENA_DIMS[1]
    ):
        return {"snake": snake, "apples": apples, "alive": False, "won": False}

    snake.append(new_coordinate)

    if new_coordinate in apples:
        apples.pop(apples.index(new_coordinate))
        apples.append(placeNewApple(snake, apples))

        return {"snake": snake, "apples": apples + 1, "alive": True, "won": False}

    snake.pop(0)  # Only remove tail coord if apple not collected

    return {"snake": snake, "apples": apples, "alive": True, "won": False}


def placeNewApple(snake, apples):
    new_apple = (randint(0, ARENA_DIMS[0]), randint(0, ARENA_DIMS[1]))
    while new_apple in snake or new_apple in apples:
        new_apple = (randint(0, ARENA_DIMS[0]), randint(0, ARENA_DIMS[1]))
    return new_apple


def gameLoop():
    snake = [(ARENA_DIMS[0] // 2, ARENA_DIMS[1] // 2)]
    apples = [placeNewApple(snake, apples)]
    direction = None

    while True:
        direction = getNewMove(snake, apples, direction)

        response = doMove(snake, apples, direction)
        if response["won"]:
            return True, len(snake)  # won, score
        if not response["alive"]:
            return False, len(snake)  # lost, score

        snake = response["snake"]
        apples = response["apples"]


def getNewMove(snake, apples, direction):
    return direction
