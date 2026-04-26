import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import os
import numpy as np
import matplotlib.pyplot as plt

from snakeGame import Environment
from ActorCritic import ActorCriticNet

MIN_STEPS = 200  # If a game ends before an iteration reaches this many steps, a new game will be started
ITERATIONS = 1800  # Number of iterations that the model trains for
PPO_EPOCHS = 5  # The number of epochs that PPO trains for
ALPHA = 0.0003  # Optimizer learning rate - Step size for updating network weights during training
GAMMA = 0.99  # Discount factor - Determines how much future rewards are valued compared to immediate rewards
LAMBD = 0.96  # GAE Lambda - Balances bias and variance in advantage estimation using Generalized Advantage Estimation
EPSILON_CLIP = 0.2  # Clipping range - Controls how much the new policy can deviate from the old one ensuring stable updates


ENTROPY_COEFFICIENT = (
    0.03  # Encourages exploration by penalizing low entropy i.e. overconfident policies
)
CRITIC_COEFFICIENT = 0.5  # Weight given to the critic loss in the total objective
BATCH_SIZE = 20  # Number of samples per update affecting stability and efficiency


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
        self,
        n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        lambd=LAMBD,
        clip_epsilon=EPSILON_CLIP,
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
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        v, probabilities = self.actor_critic(state)

        # Prevent crashes from invalid sampling (has happened before)
        try:
            action_probabilities = tfp.distributions.Categorical(probs=probabilities)
            action = action_probabilities.sample()
            self.action = action

            log_prob = action_probabilities.log_prob(action)

        except Exception as e:
            print("!! Distribution error:", e)
            action = tf.constant([np.random.randint(0, 3)])
            self.action = action

            log_prob = tf.constant([0.0])

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
        entropy = dist.entropy()

        values = tf.squeeze(values)

        return values, log_probs, entropy

    def compute_GAE_and_returns(self, iteration):
        rewards = iteration["rewards"]
        dones = iteration["dones"]
        vs = iteration["vs"]

        A = 0
        advantages = []
        for step_i in reversed(range(len(rewards))):
            # Calculate temporal difference
            delta = (
                rewards[step_i]
                + (self.gamma * vs[step_i + 1] * (1 - int(dones[step_i])))
                - vs[step_i]
            )

            # Calculate advantages (actual GAE)
            A = delta + (self.gamma * self.lambd * A * (1 - int(dones[step_i])))
            advantages.append(A)

        advantages = list(reversed(advantages))  # Flip back and the end

        # Calculate returns
        returns = np.array(advantages) + np.array(vs[:-1])

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        return advantages, returns

    def PPO_update(self, advantages, returns, iteration, mini_batch_size=BATCH_SIZE):
        # Convert to tensors and then make constant
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        advantages = tf.stop_gradient(advantages)
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + 1e-8
        )

        returns = tf.stop_gradient(tf.convert_to_tensor(returns, dtype=tf.float32))

        old_log_probs = tf.stop_gradient(
            tf.convert_to_tensor(iteration["log_probs"], dtype=tf.float32)
        )
        old_log_probs = tf.reshape(old_log_probs, [-1])

        for _ in range(PPO_EPOCHS):
            # Turn data into random set of mini-batches
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
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in dataset:
                with tf.GradientTape() as tape:
                    # Calculate the values and log probs of this epoch
                    values, new_log_probs, entropy = self.get_v_and_log_probs(
                        batch_states, batch_actions
                    )
                    values = tf.squeeze(values)

                    # Calculate PPO ratio
                    new_log_probs = tf.reshape(new_log_probs, [-1])
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

                    entropy_loss = tf.reduce_mean(entropy)

                    # Compute total loss
                    total_loss = (
                        actor_loss
                        + (CRITIC_COEFFICIENT * critic_loss)
                        - (ENTROPY_COEFFICIENT * entropy_loss)
                    )

                # Find loss gradient and back-propogate
                grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
                self.actor_critic.optimizer.apply_gradients(
                    zip(grads, self.actor_critic.trainable_variables)
                )


def get_save_files():
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

    return score_figure_file, apple_figure_file


if __name__ == "__main__":
    print("Starting training")
    env = Environment()
    agent = Agent(env.n_actions)

    score_figure_file, apple_figure_file = get_save_files()

    best_score = -999999999
    score_history = []
    apples_history = []

    for i in range(ITERATIONS):
        env.reset()
        observation = env.extractObservation()
        score = 0
        apples = 0
        deaths = 0
        done = False
        tick = 0

        iteration = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "vs": [],
        }

        # "gameloop"
        while tick < MIN_STEPS or not done:
            action, v, log_prob = agent.choose_action(observation)
            reward, done = env.doMove(action - 1)

            observation_ = env.extractObservation()
            score += reward

            iteration["states"].append(observation)
            iteration["actions"].append(action)
            iteration["rewards"].append(reward)
            iteration["dones"].append(done)
            iteration["log_probs"].append(log_prob[0].numpy())
            iteration["vs"].append(v.numpy()[0, 0])

            observation = observation_

            apples += int(env.got_apple)
            if done:
                deaths += 1
                env.reset()

            tick += 1

        iteration["states"].append(observation)
        v, _ = agent.actor_critic(
            tf.convert_to_tensor([observation], dtype=tf.float32)
        )  # evaluate final state
        iteration["vs"].append(v.numpy()[0, 0])

        advantages, returns = agent.compute_GAE_and_returns(iteration)

        agent.PPO_update(advantages, returns, iteration)

        # Plotting and benchmarking
        print(
            f"Iteration: {i} | Score: {score:.2f} | Deaths: {deaths} | Apples: {apples} | Steps: {tick}"
        )
        apples_history.append(apples)
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score:
            best_score = avg_score
            last_save = i
            agent.save_models()

    print(f"Last save at {last_save}")

    plot_learning_curve(score_history, score_figure_file, 50, "score")
    plot_learning_curve(apples_history, apple_figure_file, 50, "apples")
