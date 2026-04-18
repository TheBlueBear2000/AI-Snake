import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorCriticNet(keras.Model):
    def __init__(
        self,
        n_actions,
        fc1_dims=1024,
        fc2_dims=512,
        name="actor-critic",
        checkpoint_dir="checkpoints/actor-critic",
    ):
        super(ActorCriticNet, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".weights.h5")

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.v = Dense(1, activation=None)
        self.probabilities = Dense(n_actions, activation="softmax")

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        probabilities = self.probabilities(value)

        return v, probabilities
