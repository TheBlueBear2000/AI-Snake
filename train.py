import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from snakeGame import Environment
from ActorCritic import ActorCriticNet

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 50 scores")
    plt.savefig(figure_file)


class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=3):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNet(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probabilities = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probabilities)
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

    def save_models(self):
        print("... Saving Model ...")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print("... Loading Model ...")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            state_value, probabilities = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probabilities)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(
            zip(gradient, self.actor_critic.trainable_variables)
        )


if __name__ == "__main__":
    print("Starting training")
    env = Environment()
    agent = Agent(alpha=1e-5, n_actions=env.n_actions)
    n_games = 1800
    MAX_TICKS = 10000

    figure_file = "plots/actor-critic.png"

    best_score = -999999999
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        env.reset()
        observation = env.extractObservation()
        done = False
        score = 0

        # "gameloop"
        tick = 0
        while not done:
            action = agent.choose_action(observation)
            reward, done = env.doMove(action - 1)  # Action is 0-2, function takes -1-1

            observation_ = env.extractObservation()
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_

            tick += 1
            if tick >= MAX_TICKS:
                break

        print(f"Game {i} score: {score}")
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score:
            best_score = avg_score
            last_save = i
            if not load_checkpoint:
                agent.save_models()

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

    print(f"Last save at {last_save}")
