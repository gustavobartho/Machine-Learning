from tensorflow import random_uniform_initializer
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class Actor(object):
    def __init__(self, inp_dim, fc1_dim, fc2_dim, fc3_dim, out_dim, act_range, lr, tau):
        # Network dimensions
        self.inp_dim = inp_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.fc3_dim = fc3_dim
        self.out_dim = out_dim
        # Range of the action space
        self.act_range = act_range
        # Parameter that coordinates the soft updates on the target weights
        self.tau = tau
        # Optimizer learning rate
        self.lr = lr
        # Generates the optimization function
        self.optimizer = Adam(learning_rate=self.lr)
        # Generates the actor model
        self.model = self.buildNetwork()
        # Generates the actor target model
        self.target_model = self.buildNetwork()
        # Set the weights to be the same in the begining
        self.target_model.set_weights(self.model.get_weights())

    # --------------------------------------------------------------------
    def buildNetwork(self):
        inp = Input(shape=(self.inp_dim,))

        f1 = 1 / np.sqrt(self.fc1_dim)
        fc1 = Dense(self.fc1_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f1, f1), bias_initializer=random_uniform_initializer(-f1, f1), dtype='float64')(inp)
        norm1 = BatchNormalization(dtype='float64')(fc1)

        f2 = 1 / np.sqrt(self.fc2_dim)
        fc2 = Dense(self.fc2_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f2, f2), bias_initializer=random_uniform_initializer(-f2, f2), dtype='float64')(norm1)
        norm2 = BatchNormalization(dtype='float64')(fc2)

        f3 = 1 / np.sqrt(self.fc3_dim)
        fc3 = Dense(self.fc3_dim, activation='relu', kernel_initializer=random_uniform_initializer(
            -f3, f3), bias_initializer=random_uniform_initializer(-f3, f3), dtype='float64')(norm2)
        norm3 = BatchNormalization(dtype='float64')(fc3)

        f3 = 0.003
        out = Dense(self.out_dim, activation='tanh', kernel_initializer=random_uniform_initializer(
            -f3, f3), bias_initializer=random_uniform_initializer(-f3, f3), dtype='float64')(norm3)
        lamb = Lambda(lambda i: i * self.act_range, dtype='float64')(out)

        return Model(inputs=[inp], outputs=[lamb])

    # --------------------------------------------------------------------
    def predict(self, states):
        return self.model([states], training=False)

    # --------------------------------------------------------------------
    def target_predict(self, states):
        return self.target_model([states], training=False)

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
        self.model.save_weights(path + '_actor.h5')

    # --------------------------------------------------------------------
    def loadModel(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
