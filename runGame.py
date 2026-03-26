from random import randint
from time import sleep

RENDER_FPS = 2


def doMove(snake, apples, direction, arena_dims):
    if direction == None:
        return {"snake": snake, "apples": apples, "alive": True, "won": False}

    if len(snake) == arena_dims[0] * arena_dims[1]:
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
        or new_coordinate[0] >= arena_dims[0]
        or new_coordinate[1] < 0
        or new_coordinate[1] >= arena_dims[1]
    ):
        return {"snake": snake, "apples": apples, "alive": False, "won": False}

    snake.append(new_coordinate)

    if new_coordinate in apples:
        apples.pop(apples.index(new_coordinate))
        apples.append(placeNewApple(snake, apples, arena_dims))

        return {"snake": snake, "apples": apples + 1, "alive": True, "won": False}

    snake.pop(0)  # Only remove tail coord if apple not collected

    return {"snake": snake, "apples": apples, "alive": True, "won": False}


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


def gameLoop(rendering=False, arena_dims=(20, 15)):
    snake = [(arena_dims[0] // 2, arena_dims[1] // 2)]
    apples = [placeNewApple(snake, [], arena_dims)]
    direction = "up"

    while True:
        direction = getNewMove(snake, apples, direction)

        response = doMove(snake, apples, direction, arena_dims)
        if response["won"]:
            return True, len(snake)  # won, score
        if not response["alive"]:
            return False, len(snake)  # lost, score

        snake = response["snake"]
        apples = response["apples"]

        if rendering:
            render(snake, apples, arena_dims)
            sleep(1 / RENDER_FPS)


def getNewMove(snake, apples, direction):
    return "up"


if __name__ == "__main__":
    gameLoop(rendering=True)
