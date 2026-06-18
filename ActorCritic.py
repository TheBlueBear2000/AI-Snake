import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate


class ActorCriticNet(keras.Model):
    def __init__(
        self,
        n_actions,
        name="actor-critic",
        checkpoint_dir="checkpoints/actor-critic",
    ):
        super(ActorCriticNet, self).__init__()

        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".weights.h5")

        # CNN for board
        self.conv1 = Conv2D(32, 3, activation="relu", padding="same")
        self.conv2 = Conv2D(64, 3, activation="relu", padding="same")
        self.flatten = Flatten()

        # Standard feature scalar
        self.fc_meta1 = Dense(64, activation="relu")
        self.fc_meta2 = Dense(64, activation="relu")

        # Combined trunk
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(256, activation="relu")

        # Heads
        self.v = Dense(1, activation=None)
        self.probabilities = Dense(n_actions, activation="softmax")

    def call(self, observation):
        meta, board = observation

        # CNN branch
        x = self.conv1(board)
        x = self.conv2(x)
        x = self.flatten(x)

        # MLP branch
        m = self.fc_meta1(meta)
        m = self.fc_meta2(m)

        # Merge
        combined = Concatenate()([x, m])

        # Shared trunk
        value = self.fc1(combined)
        value = self.fc2(value)

        v = self.v(value)
        probabilities = self.probabilities(value)

        return v, probabilities
