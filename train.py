import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import os

from snakeGame import Environment
from ActorCritic import ActorCriticNet

import numpy as np
import matplotlib.pyplot as plt

GAME_STEPS = 200
ITERATIONS = 1800
PPO_EPOCHS = 15


def plot_learning_curve(values, figure_file, number, name):
    plt.figure()

    x = [i + 1 for i in range(len(values))]

    running_avg = np.zeros(len(values))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(values[max(0, i - number) : (i + 1)])

    plt.plot(x, running_avg)
    plt.title(f"Running average of previous {number} {name}")
    plt.savefig(figure_file)
    plt.close()


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

    def get_v_and_log_probs(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        v, probs = self.actor_critic(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        log_probs = action_probs.log_prob(self.action_space)
        return v, log_probs

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

    current_file = 0
    files = os.listdir("plots/")
    for file_name in files:
        file_name = file_name.replace("actor-critic-score_", "")
        file_name = file_name.replace("actor-critic-apples_", "")
        file_name = file_name.replace(".png", "")
        if file_name.isnumeric():
            current_file = max(current_file, int(file_name))
    current_file += 1
    print(f"Saving to plot number {current_file}")

    score_figure_file = f"plots/actor-critic-score_{current_file}.png"
    apple_figure_file = f"plots/actor-critic-apples_{current_file}.png"

    best_score = -999999999
    score_history = []
    apples_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(ITERATIONS):
        env.reset()
        observation = env.extractObservation()
        score = 0
        apples = 0
        deaths = 0

        iteration = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "vs": [],
        }

        # "gameloop"
        for tick in range(GAME_STEPS):
            action = agent.choose_action(observation)
            reward, done = env.doMove(action - 1)

            apples += int(env.got_apple)

            observation_ = env.extractObservation()
            score += reward

            iteration["states"].append(observation)
            iteration["actions"].append(action)
            iteration["rewards"].append(reward)
            iteration["dones"].append(done)

            v, log_probs = agent.get_v_and_log_probs(observation)
            iteration["log_probs"].append(log_probs)
            iteration["vs"].append(v)

            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_

            if done:
                deaths += 1
                env.reset()

        iteration["states"].append(observation)

        print(
            f"Iteration: {i} | Score: {score:.2f} | Deaths: {deaths} | Apples: {apples}"
        )
        apples_history.append(apples)
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score:
            best_score = avg_score
            last_save = i
            if not load_checkpoint:
                agent.save_models()

    print(f"Last save at {last_save}")

    plot_learning_curve(score_history, score_figure_file, 50, "score")
    plot_learning_curve(apples_history, apple_figure_file, 50, "apples")
