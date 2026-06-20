import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, LSTM


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
        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(128, activation="relu")
        self.lstm = LSTM(128, return_sequences=False)

        # Heads
        self.v = Dense(1, activation=None)
        self.probabilities = Dense(n_actions, activation="softmax")

    def call(self, meta, board):

        # board: (batch, time, H, W, C)
        # meta : (batch, time, features)

        B = tf.shape(board)[0]
        T = tf.shape(board)[1]

        # Merge batch and time so CNN sees ordinary images
        x = tf.reshape(board, (-1, board.shape[2], board.shape[3], board.shape[4]))

        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)

        # Metadata
        m = tf.reshape(meta, (-1, meta.shape[2]))
        m = self.fc_meta1(m)
        m = self.fc_meta2(m)

        # Combined Trunk
        combined = tf.concat([x, m], axis=-1)
        combined = self.fc1(combined)

        # Restore time dimension
        combined = tf.reshape(combined, (B, T, -1))

        hidden = self.lstm(combined)

        hidden = self.fc2(hidden)

        value = self.v(hidden)
        probs = self.probabilities(hidden)

        return value, probs
