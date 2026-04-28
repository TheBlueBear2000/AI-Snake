from random import randint
from time import sleep
from math import sqrt
from enum import Enum

RENDER_FPS = 2
MAX_APPLES = 3
START_LENGTH = 3
APPLE_LENGTH_BONUS = 2


class Directions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Environment:
    def __init__(self):
        self.n_actions = 3
        self.arena_dims = (20, 15)
        self.reset()

    def reset(self):
        self.snake = [(self.arena_dims[0] // 2, self.arena_dims[1] // 2)]
        self.snake_backlog = START_LENGTH - 1
        self.apples = []
        self.placeNewApples()
        self.direction = Directions.UP
        self.steps_since_apple = -1
        self.got_apple = False
        self.last_dist = 10000

    def doMove(self, move):
        self.got_apple = False
        if move > 1 or move < -1:
            raise Exception(f"Moves must be a value of -1, 0 or 1, not {move}")

        if len(self.snake) == self.arena_dims[0] * self.arena_dims[1]:
            # Won
            print("!! Won game !!")
            return 10000, True  # reward, done

        # moves are -1 = left, 0 = forward, 1 = right
        self.direction = Directions((self.direction.value + move) % 4)

        if self.direction == Directions.UP:
            new_coordinate = (self.snake[-1][0], self.snake[-1][1] - 1)
        elif self.direction == Directions.DOWN:
            new_coordinate = (self.snake[-1][0], self.snake[-1][1] + 1)
        elif self.direction == Directions.LEFT:
            new_coordinate = (self.snake[-1][0] - 1, self.snake[-1][1])
        elif self.direction == Directions.RIGHT:
            new_coordinate = (self.snake[-1][0] + 1, self.snake[-1][1])

        if (
            new_coordinate[0] < 0
            or new_coordinate[0] >= self.arena_dims[0]
            or new_coordinate[1] < 0
            or new_coordinate[1] >= self.arena_dims[1]
            or new_coordinate in self.snake
        ):
            # Died by hitting wall
            return -50, True  # reward, done

        self.snake.append(new_coordinate)

        if new_coordinate in self.apples:
            self.apples.pop(self.apples.index(new_coordinate))
            self.placeNewApples()
            self.snake_backlog = APPLE_LENGTH_BONUS - 1
            self.steps_since_apple = -1
            self.got_apple = True
            # Apple
            return 10, False  # reward, done

        if self.snake_backlog > 0:
            self.snake_backlog -= 1
        else:
            self.snake.pop(0)  # Only remove tail coord if apple not collected

        # Living reward
        head = self.snake[-1]
        nearest_apple = self.calculateNearestApple()

        current_dist = ((head[0] - nearest_apple[0]) ** 2) + (
            (head[1] - nearest_apple[1]) ** 2
        )
        if current_dist < self.last_dist:
            living_reward = 0.1
        else:
            living_reward = -0.15
        self.last_dist = current_dist

        return living_reward, False  # reward, done

    def placeNewApples(self):
        while len(self.apples) < min(
            MAX_APPLES, (self.arena_dims[0] * self.arena_dims[1]) - len(self.snake)
        ):
            new_apple = (randint(0, self.arena_dims[0]), randint(0, self.arena_dims[1]))
            while new_apple in self.snake or new_apple in self.apples:
                new_apple = (
                    randint(0, self.arena_dims[0] - 1),
                    randint(0, self.arena_dims[1] - 1),
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

        # values 0 & 1 (food direction)
        head = self.snake[-1]
        nearest_apple = self.calculateNearestApple()
        above = head[1] - nearest_apple[1]
        left = head[0] - nearest_apple[0]

        # values must be in order: ahead, to the side
        if self.direction == Directions.UP:
            values += [above, left]
        elif self.direction == Directions.DOWN:
            values += [above * -1, left * -1]
        elif self.direction == Directions.LEFT:
            values += [left, above * -1]
        elif self.direction == Directions.RIGHT:
            values += [left * -1, above]

        # values 2-4 (immediate danger)
        changes = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # Iterates from starting point depending on direction
        for i in list(range(self.direction.value, len(changes))) + list(
            range(0, self.direction.value)
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
        values.pop(-2)  # Remove the check directly behind

        # values 5-11 (surrounding danger)
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
        for i in list(range(self.direction.value * 2, len(changes))) + list(
            range(0, self.direction.value * 2)
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
        values.pop(-4)  # Remove the check directly behind

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


if __name__ == "__main__":
    from train import Agent

    env = Environment()
    agent = Agent(env.n_actions)

    # Do dummy observation to demonstrate shape of modle to tf before loading
    observation = env.extractObservation()
    agent.choose_action(observation)
    agent.load_models()

    done = False
    while not done:
        env.render()

        action, _, _ = agent.choose_action(observation)
        observation = env.extractObservation()

        _, done = env.doMove(action - 1)

        sleep(1 / RENDER_FPS)
