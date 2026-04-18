from random import randint
from time import sleep
from math import sqrt

RENDER_FPS = 2


def doMove(snake, apples, direction, arena_dims):
    if direction == None:
        return {
            "snake": snake,
            "apples": apples,
            "alive": True,
            "won": False,
            "reward": 0,
        }

    if len(snake) == arena_dims[0] * arena_dims[1]:  # Won
        return {
            "snake": snake,
            "apples": apples,
            "alive": True,
            "won": True,
            "reward": 500,
        }

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
        or new_coordinate[0] >= arena_dims[0]
        or new_coordinate[1] < 0
        or new_coordinate[1] >= arena_dims[1]
    ):
        return {
            "snake": snake,
            "apples": apples,
            "alive": False,
            "won": False,
            "reward": -20,
        }

    snake.append(new_coordinate)

    if new_coordinate in apples:
        apples.pop(apples.index(new_coordinate))
        apples.append(placeNewApple(snake, apples, arena_dims))

        return {
            "snake": snake,
            "apples": apples + 1,
            "alive": True,
            "won": False,
            "reward": 10,
        }

    snake.pop(0)  # Only remove tail coord if apple not collected

    return {
        "snake": snake,
        "apples": apples,
        "alive": True,
        "won": False,
        "reward": -0.1,
    }


def placeNewApple(snake, apples, arena_dims):
    new_apple = (randint(0, arena_dims[0]), randint(0, arena_dims[1]))
    while new_apple in snake or new_apple in apples:
        new_apple = (randint(0, arena_dims[0]), randint(0, arena_dims[1]))
    return new_apple


def render(snake, apples, arena_dims):
    out = "+" + (arena_dims[0] * "--") + "+\n|"
    for y in range(arena_dims[1]):
        for x in range(arena_dims[0]):
            if (x, y) in snake:
                out += "\033[0;32m[]\033[0;50m"  # green
            elif (x, y) in apples:
                out += "\033[0;31m[]\033[0;50m"  # red
            else:
                out += "  "
        out += "|\n"
        if y < arena_dims[1] - 1:
            out += "|"
    out += "+" + (arena_dims[0] * "--") + "+"
    print(out)


def extractNetValues(direction, snake, apples, arena_dims):
    values = []
    # values 0-3 (current direction)
    if direction == "up":
        values += [1, 0, 0, 0]
    if direction == "down":
        values += [0, 1, 0, 0]
    if direction == "left":
        values += [0, 0, 1, 0]
    if direction == "right":
        values += [0, 0, 0, 1]

    # values 4-7 (food direction)
    head = snake[-1]
    nearest_apple = calculateNearestApple(apples, head)
    above = head[1] - nearest_apple[1] > 0
    left = head[0] - nearest_apple[0] > 0
    values += [int(above), int(not above), int(left), int(not left)]

    # values 8-11 (immediate danger)
    for change in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        new_coordinate = (head[0] + change[0], head[1] + change[1])
        values.append(
            int(
                new_coordinate in snake
                or new_coordinate[0] < 0
                or new_coordinate[0] >= arena_dims[0]
                or new_coordinate[1] < 0
                or new_coordinate[1] >= arena_dims[1]
            )
        )

    # values 12-19 (surrounding danger)
    for change in [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]:
        start = True
        step = 0
        while start or not (
            new_coordinate in snake
            or new_coordinate[0] < 0
            or new_coordinate[0] >= arena_dims[0]
            or new_coordinate[1] < 0
            or new_coordinate[1] >= arena_dims[1]
        ):
            start = False
            step += 1
            new_coordinate = (
                head[0] + (change[0] * step),
                head[1] + (change[1] * step),
            )
        values.append(steps)
    return values


def calculateNearestApple(apples, head):
    nearest = None
    min_dist = 999999999
    for apple in apples:
        dist = sqrt(((apple[0] - head[0]) ** 2) + ((apple[1] - head[1]) ** 2))
        if dist < min_dist:
            nearest = apple
            min_dist = dist
    return nearest


def gameLoop(rendering=False, arena_dims=(20, 15), steps=None, startStep=0):
    snake = [(arena_dims[0] // 2, arena_dims[1] // 2)]
    apples = [placeNewApple(snake, [], arena_dims)]
    direction = "up"
    values = []
    rewards = []

    iterations = 0
    while True:
        direction = getNewMove(snake, apples, direction)
        values.append(extractNetValues(direction, snake, apples, arena_dims))

        response = doMove(snake, apples, direction, arena_dims)
        rewards.append(response["reward"])

        if response["won"]:
            return (
                iterations,
                values,
                rewards,
                extractNetValues(direction, snake, apples, arena_dims),
            )  # iterations, values, rewards, next_val
        if not response["alive"]:
            return (
                iterations,
                values,
                rewards,
                extractNetValues(direction, snake, apples, arena_dims),
            )  # iterations, values, rewards, next_val

        snake = response["snake"]
        apples = response["apples"]

        if rendering:
            render(snake, apples, arena_dims)
            sleep(1 / RENDER_FPS)

        if steps != None and iterations < steps - startStep:
            break
        iterations += 1
    return (
        iterations,
        values,
        rewards,
        extractNetValues(direction, snake, apples, arena_dims),
    )  # iterations, values, rewards, next_val


def getNewMove(snake, apples, direction):
    return "up"


if __name__ == "__main__":
    gameLoop(rendering=True)
