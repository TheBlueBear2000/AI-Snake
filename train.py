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
    def __init__(
        self, alpha=0.0003, gamma=0.99, lambd=0.96, clip_epsilon=0.2, n_actions=3
    ):
        self.gamma = gamma
        self.lambd = lambd
        self.clip_epsilon = clip_epsilon
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNet(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        v, probabilities = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probabilities)
        action = action_probabilities.sample()
        self.action = action

        log_prob = action_probabilities.log_prob(action)

        return action.numpy()[0], v, log_prob

    def save_models(self):
        print("... Saving Model ...")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print("... Loading Model ...")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def get_v_and_log_probs(self, states, actions):
        states = tf.cast(states, tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        values, probs = self.actor_critic(states)

        dist = tfp.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)

        values = tf.squeeze(values)

        return values, log_probs

    def compute_GAE_and_returns(self, iteration):
        rewards = iteration["rewards"]
        dones = iteration["dones"]
        vs = iteration["vs"]

        A = 0
        As = []
        Rs = []
        for step_i in reversed(range(len(rewards))):
            # Calculate temporal difference
            delta = (
                rewards[step_i]
                + (self.gamma * vs[step_i + 1] * (1 - int(dones[step_i])))
                - vs[step_i]
            )

            # Calculate advantages (actual GAE)
            A = delta + (self.gamma * self.lambd * A * (1 - int(dones[step_i])))
            As.append(A)

            # Calculate return
            R = A + vs[step_i]
            Rs.append(R)

        return list(reversed(As)), list(reversed(Rs))  # Flip back and the end

    def PPO_update(self, advantages, returns, iteration, mini_batch_size=20):
        # Convert to tensors and then make constant
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + 1e-8
        )
        advantages = tf.stop_gradient(advantages)

        returns = tf.stop_gradient(tf.convert_to_tensor(returns, dtype=tf.float32))

        old_log_probs = tf.stop_gradient(
            tf.convert_to_tensor(iteration["log_probs"], dtype=tf.float32)
        )

        # Turn data into set of mini-batches
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                (
                    iteration["states"][:-1],
                    iteration["actions"],
                    old_log_probs,
                    returns,
                    advantages,
                )
            )
            .shuffle(len(iteration["rewards"]))
            .batch(mini_batch_size)
        )

        for _ in range(PPO_EPOCHS):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in dataset:
                with tf.GradientTape() as tape:
                    # Calculate the values and log probs of this epoch
                    values, new_log_probs = self.get_v_and_log_probs(
                        batch_states, batch_actions
                    )
                    values = tf.squeeze(values)

                    # Calculate PPO ratio
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)

                    # Apply clipping
                    clipped_ratio = tf.clip_by_value(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    )

                    actor_loss = -tf.reduce_mean(
                        tf.minimum(
                            ratio * batch_advantages, clipped_ratio * batch_advantages
                        )
                    )
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values))

                    total_loss = actor_loss + (
                        0.5 * critic_loss
                    )  # - (0.01 * entropy) for when I implement entropy

                # Find loss gradient and back-propogate
                grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
                self.actor_critic.optimizer.apply_gradients(
                    zip(grads, self.actor_critic.trainable_variables)
                )

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
            action, v, log_prob = agent.choose_action(observation)
            reward, done = env.doMove(action - 1)

            apples += int(env.got_apple)

            observation_ = env.extractObservation()
            score += reward

            iteration["states"].append(observation)
            iteration["actions"].append(action)
            iteration["rewards"].append(reward)
            iteration["dones"].append(done)
            iteration["log_probs"].append(log_prob)
            iteration["vs"].append(v)

            # if not load_checkpoint:
            #    agent.learn(observation, reward, observation_, done)
            observation = observation_

            if done:
                deaths += 1
                env.reset()

        iteration["states"].append(observation)
        v, _ = agent.actor_critic(
            tf.convert_to_tensor([observation])
        )  # evaluate final state
        iteration["vs"].append(v)

        advantages, returns = agent.compute_GAE_and_returns(iteration)

        agent.PPO_update(advantages, returns, iteration)

        # Plotting and benchmarking
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
