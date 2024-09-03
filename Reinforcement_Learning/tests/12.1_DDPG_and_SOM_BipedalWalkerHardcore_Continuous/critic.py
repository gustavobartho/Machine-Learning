from tensorflow import random_uniform_initializer
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class Critic(object):
    def __init__(
        self,
        state_inp_dim,
        state_fc1_dim,
        state_fc2_dim,
        action_inp_dim,
        action_fc1_dim,
        conc_fc1_dim,
        conc_fc2_dim,
        out_dim,
        lr,
        tau,
    ):

        # Network dimensions
        self.state_inp_dim = state_inp_dim
        self.state_fc1_dim = state_fc1_dim
        self.state_fc2_dim = state_fc2_dim
        self.action_inp_dim = action_inp_dim
        self.action_fc1_dim = action_fc1_dim
        self.conc_fc1_dim = conc_fc1_dim
        self.conc_fc2_dim = conc_fc2_dim
        self.out_dim = out_dim

        # Optimizer learning rate
        self.lr = lr

        # Define the critic optimizer
        self.optimizer = Adam(learning_rate=self.lr)

        # Parameter that coordinates the soft updates on the target weights
        self.tau = tau

        # Generate the critic network
        self.model = self.buildNetwork()

        # Generate the critic target network
        self.target_model = self.buildNetwork()

        # Set the weights to be the same in the begining
        self.target_model.set_weights(self.model.get_weights())

    # --------------------------------------------------------------------
    def buildNetwork(self):
        # State input network ---------
        s_inp = Input(shape=(self.state_inp_dim, ))

        f1 = 1 / np.sqrt(self.state_fc1_dim)
        s_fc1 = Dense(self.state_fc1_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f1, f1), bias_initializer=random_uniform_initializer(-f1, f1), dtype='float64')(s_inp)
        s_norm1 = BatchNormalization(dtype='float64')(s_fc1)

        f2 = 1 / np.sqrt(self.state_fc2_dim)
        s_fc2 = Dense(self.state_fc2_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f2, f2), bias_initializer=random_uniform_initializer(-f2, f2), dtype='float64')(s_norm1)
        s_norm2 = BatchNormalization(dtype='float64')(s_fc2)

        # Action input network ---------
        a_inp = Input(shape=(self.action_inp_dim, ))

        f1 = 1 / np.sqrt(self.action_fc1_dim)
        a_fc1 = Dense(self.action_fc1_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f1, f1), bias_initializer=random_uniform_initializer(-f1, f1), dtype='float64')(a_inp)
        a_norm1 = BatchNormalization(dtype='float64')(a_fc1)

        # Concatenate the two networks ---
        c_inp = Concatenate(dtype='float64')([s_norm2, a_norm1])

        # Creates the output network
        f1 = 1 / np.sqrt(self.conc_fc1_dim)
        c_fc1 = Dense(self.conc_fc1_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f1, f1), bias_initializer=random_uniform_initializer(-f1, f1), dtype='float64')(c_inp)
        c_norm1 = BatchNormalization(dtype='float64')(c_fc1)

        f2 = 1 / np.sqrt(self.conc_fc2_dim)
        c_fc2 = Dense(self.conc_fc2_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f2, f2), bias_initializer=random_uniform_initializer(-f2, f2), dtype='float64')(c_norm1)
        c_norm2 = BatchNormalization(dtype='float64')(c_fc2)

        f3 = 0.003
        out = Dense(self.out_dim, activation='linear', kernel_initializer=random_uniform_initializer(
            -f3, f3), bias_initializer=random_uniform_initializer(-f3, f3), dtype='float64')(c_norm2)

        model = Model(inputs=[s_inp, a_inp], outputs=[out])

        return model

    # --------------------------------------------------------------------
    def predict(self, states, actions):
        return self.model([states, actions], training=False)

    # --------------------------------------------------------------------
    def target_predict(self, states, actions):
        return self.target_model([states, actions], training=False)

    # --------------------------------------------------------------------
    def transferWeights(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = []

        for i in range(len(weights)):
            new_weights.append(
                (self.tau * weights[i]) + ((1.0 - self.tau) * target_weights[i]))

        self.target_model.set_weights(new_weights)

    # --------------------------------------------------------------------
    def saveModel(self, path):
        self.model.save_weights(path + '_critic.h5')

    # --------------------------------------------------------------------
    def loadModel(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
