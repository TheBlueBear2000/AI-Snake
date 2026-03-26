# AI-Snake
This project aims to play a game of snake using an artificially intelligent model. The model uses a neural network (outlined below) to play the game, and is improved through RL (Reinforcement Learning), using a reward function also outlined below.

# Neural Network Architecture
## Input Features
| Number | Name | Description |
| --- | --- | --- |
| 0-3 | Direction | The direction of the snake (0 is up, 1 is down, 2 is left, 3 is right) |
| 4-7 | Food Direction | The direction of the nearest piece of food (4 is up, 5 is down, 6 is left, 7 is right) |
| 8-11 | Safety | Directly adjacent objects (8 is up, 9 is down, 10 is left, 11 is right) |
| 12-19 | Range | Distances to collidables in 45 degree increments (12 is up, then iterativley clockwise) |

## Output Features
| Number | Name | Description |
| --- | --- | --- |
| 0-3 | Direction | The newly chosen direction of the snake (0 is up, 1 is down, 2 is left, 3 is right) |

There are two layers, of 256 and then 128 neurons

A ReLU activation function is used

# Training
To train the model, it is run for a full game and then it's performance is used to adjust the it's parameters

## Reward Function
| Name | Gain/Loss (+/-) | Description |
| --- | --- | --- |
| Score | +10 | Reward for each snake length / apples collected / game score |
| Iterations | -0.1 | Penalty for how long the game lasted, per iteration |
| Win | +500 | Massive reward for winning game |



