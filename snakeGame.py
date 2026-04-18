from random import randint
from time import sleep
from math import sqrt

RENDER_FPS = 2
DIRECTIONS = ["up", "right", "down", "left"]
MAX_APPLES = 3


class Environment:
    def __init__(self):
        self.n_actions = 3
        self.arena_dims = (20, 15)
        self.snake = [(self.arena_dims[0] // 2, self.arena_dims[1] // 2)]
        self.apples = []
        self.placeNewApples()
        self.direction = "up"

    def reset(self):
        self.snake = [(self.arena_dims[0] // 2, self.arena_dims[1] // 2)]
        self.apples = []
        self.placeNewApples()
        self.direction = "up"

    def doMove(self, move):
        if move > 1 or move < -1:
            raise Exception(f"Moves must be a value of -1, 0 or 1, not {move}")

        if len(self.snake) == self.arena_dims[0] * self.arena_dims[1]:
            # Won
            print("Won game!!")
            return 10000, True  # reward, done

        # moves are -1 = left, 0 = forward, 1 = right
        self.direction = DIRECTIONS[(DIRECTIONS.index(self.direction) + move) % 4]

        if self.direction == "up":
            new_coordinate = (self.snake[-1][0], self.snake[-1][1] - 1)
        elif self.direction == "down":
            new_coordinate = (self.snake[-1][0], self.snake[-1][1] + 1)
        elif self.direction == "left":
            new_coordinate = (self.snake[-1][0] - 1, self.snake[-1][1])
        elif self.direction == "right":
            new_coordinate = (self.snake[-1][0] + 1, self.snake[-1][1])

        if (
            new_coordinate in self.snake
            or new_coordinate[0] < 0
            or new_coordinate[0] >= self.arena_dims[0]
            or new_coordinate[1] < 0
            or new_coordinate[1] >= self.arena_dims[1]
        ):
            # Died
            print("died")
            return 0, True  # reward, done

        self.snake.append(new_coordinate)

        if new_coordinate in self.apples:
            self.apples.pop(self.apples.index(new_coordinate))
            self.placeNewApples()
            # Apple
            return 15, False  # reward, done

        self.snake.pop(0)  # Only remove tail coord if apple not collected

        # Nothing
        return -1, False  # reward, done

    def placeNewApples(self):
        while len(self.apples) < min(
            MAX_APPLES, (self.arena_dims[0] * self.arena_dims[1]) - len(self.snake)
        ):
            new_apple = (randint(0, self.arena_dims[0]), randint(0, self.arena_dims[1]))
            while new_apple in self.snake or new_apple in self.apples:
                new_apple = (
                    randint(0, self.arena_dims[0]),
                    randint(0, self.arena_dims[1]),
                )
            self.apples.append(new_apple)

    def render(self):
        out = "+" + (self.arena_dims[0] * "--") + "+\n|"
        for y in range(self.arena_dims[1]):
            for x in range(self.arena_dims[0]):
                if (x, y) in self.snake:
                    out += "\033[0;32m[]\033[0;50m"  # green
                elif (x, y) in self.apples:
                    out += "\033[0;31m[]\033[0;50m"  # red
                else:
                    out += "  "
            out += "|\n"
            if y < self.arena_dims[1] - 1:
                out += "|"
        out += "+" + (self.arena_dims[0] * "--") + "+"
        print(out)

    def extractObservation(self):
        values = []

        # values 0-3 (food direction)
        head = self.snake[-1]
        nearest_apple = self.calculateNearestApple()
        above = head[1] - nearest_apple[1] > 0
        left = head[0] - nearest_apple[0] > 0

        # values must be in order: ahead, behind, left, right of head
        if self.direction == "up":
            values += [int(above), int(not above), int(left), int(not left)]
        elif self.direction == "down":
            values += [int(not above), int(above), int(not left), int(left)]
        elif self.direction == "left":
            values += [int(left), int(not left), int(not above), int(above)]
        elif self.direction == "right":
            values += [int(not left), int(left), int(above), int(not above)]

        # values 4-7 (immediate danger)
        changes = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # Iterates from starting point depending on direction
        for i in list(range(DIRECTIONS.index(self.direction), len(changes))) + list(
            range(0, DIRECTIONS.index(self.direction))
        ):
            change = changes[i]
            new_coordinate = (head[0] + change[0], head[1] + change[1])
            values.append(
                int(
                    new_coordinate in self.snake
                    or new_coordinate[0] < 0
                    or new_coordinate[0] >= self.arena_dims[0]
                    or new_coordinate[1] < 0
                    or new_coordinate[1] >= self.arena_dims[1]
                )
            )

        # values 8-15 (surrounding danger)
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
        # Iterates from starting point depending on direction
        for i in list(range(DIRECTIONS.index(self.direction) * 2, len(changes))) + list(
            range(0, DIRECTIONS.index(self.direction) * 2)
        ):
            change = changes[i]
            start = True
            steps = 0
            while start or not (
                new_coordinate in self.snake
                or new_coordinate[0] < 0
                or new_coordinate[0] >= self.arena_dims[0]
                or new_coordinate[1] < 0
                or new_coordinate[1] >= self.arena_dims[1]
            ):
                start = False
                steps += 1
                new_coordinate = (
                    head[0] + (change[0] * steps),
                    head[1] + (change[1] * steps),
                )
            values.append(steps)
        return values

    def calculateNearestApple(self):
        head = self.snake[-1]
        nearest = None
        min_dist = 999999999
        for apple in self.apples:
            dist = sqrt(((apple[0] - head[0]) ** 2) + ((apple[1] - head[1]) ** 2))
            if dist < min_dist:
                nearest = apple
                min_dist = dist
        return nearest


def gameLoop():
    env = Environment()
    agent = Agent(alpha=1e-5, n_actions=env.n_actions)
    agent.load_models()

    done = False
    while not done:
        action = agent.choose_action(observation)
        observation = env.extractObservation()

        _, done = env.doMove(action - 1)  # Action is 0-2, function takes -1-1

        env.render()
        sleep(1 / RENDER_FPS)


if __name__ == "__main__":
    gameLoop()
