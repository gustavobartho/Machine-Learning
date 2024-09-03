from tensorflow.keras.preprocessing .image import smart_resize
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from tensorflow.math import reduce_mean, square
from tensorflow import GradientTape, random_uniform_initializer
import matplotlib.pyplot as plt
import os
import time
import gym
import numpy as np
import warnings
warnings.filterwarnings('ignore')


##################################################################

# Ornstein-Uhlenbeck Noise

class OUActionNoise(object):
    def __init__(self, mean, sigma=0.5, theta=0.4, dt=0.1, x0=None):
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    # --------------------------------------------------------------------------------
    # Method that enables to write classes where the instances behave like functions and can be called like a function.
    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mean.shape)
        self.x_prev = x

        return x

    # --------------------------------------------------------------------------------
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mean)

##################################################################

# Replay Buffer


class ReplayBuffer(object):
    def __init__(self, size, minibatch_size=None):
        '''
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
        '''
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState()
        self.max_size = size

    # --------------------------------------------------------------------------------
    def append(self, state, action, reward, next_state, done, embedding, new_embedding):
        '''
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            done (boolen): True if the next state is a terminal state and False otherwise.
                           Is transformed to integer so tha True = 1, False = 0
            next_state (Numpy array): The next state.           
        '''
        # Has a 60% chance of registering the memory
        if self.hasMin and np.random.rand() > 0.6:
            return
        if self.size() == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, next_state,
                           int(done), embedding, new_embedding])

    # --------------------------------------------------------------------------------
    def sample(self):
        '''
        Returns:
            A list of transition tuples including state, action, reward, terminal, and next_state
        '''
        idxs = self.rand_generator.choice(
            np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    # --------------------------------------------------------------------------------
    def size(self):
        '''
        Returns:
            Number of elements in the buffer
        '''
        return len(self.buffer)

    # --------------------------------------------------------------------------------
    @property
    def hasMin(self):
        '''
        Returns:
            Boolean indicating if the memory have the minimum number of elements or not
        '''
        return (self.size() >= self.minibatch_size)

    # --------------------------------------------------------------------------------
    def empties(self):
        self.buffer.clear()

    # --------------------------------------------------------------------------------
    def getEpisode(self):
        '''
        Returns:
            List with all the elements in the buffer
        '''
        return self.buffer

##################################################################


class Actor(object):
    def __init__(
        self,
        state_dim,
        state_fc1_dim,
        state_fc2_dim,
        embedding_dim,
        embedding_fc1_dim,
        embedding_fc2_dim,
        embedding_fc3_dim,
        concat_fc1_dim,
        concat_fc2_dim,
        concat_fc3_dim,
        out_dim,
        act_range,
        lr,
        tau,
    ):
        # Network dimensions
        self.state_dim = state_dim
        self.state_fc1_dim = state_fc1_dim
        self.state_fc2_dim = state_fc2_dim
        self.embedding_dim = embedding_dim
        self.embedding_fc1_dim = embedding_fc1_dim
        self.embedding_fc2_dim = embedding_fc2_dim
        self.embedding_fc3_dim = embedding_fc3_dim
        self.concat_fc1_dim = concat_fc1_dim
        self.concat_fc2_dim = concat_fc2_dim
        self.concat_fc3_dim = concat_fc3_dim
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

        return

    # --------------------------------------------------------------------

    def buildNetwork(self):

        # State network
        s_inp = Input(shape=(self.state_dim, ))

        s_fc1 = Dense(self.state_fc1_dim, activation='relu',
                      dtype='float64')(s_inp)
        s_norm1 = BatchNormalization(dtype='float64')(s_fc1)

        s_fc2 = Dense(self.state_fc2_dim, activation='relu',
                      dtype='float64')(s_norm1)
        s_norm2 = BatchNormalization(dtype='float64')(s_fc2)
        s_drop2 = Dropout(0.2)(s_norm2)

        # Embedding network
        e_inp = Input(shape=(self.embedding_dim, ))

        e_fc1 = Dense(self.embedding_fc1_dim,
                      activation='relu', dtype='float64')(e_inp)
        e_norm1 = BatchNormalization(dtype='float64')(e_fc1)

        e_fc2 = Dense(self.embedding_fc2_dim, activation='relu',
                      dtype='float64')(e_norm1)
        e_norm2 = BatchNormalization(dtype='float64')(e_fc2)
        e_drop2 = Dropout(0.3)(e_norm2)

        e_fc3 = Dense(self.embedding_fc3_dim, activation='relu',
                      dtype='float64')(e_drop2)
        e_norm3 = BatchNormalization(dtype='float64')(e_fc3)

        # Concatenate the two networks ---
        c_inp = Concatenate(dtype='float64')([s_drop2, e_norm3])

        # Creates the output network

        c_fc1 = Dense(self.concat_fc1_dim, activation='relu',
                      dtype='float64')(c_inp)
        c_norm1 = BatchNormalization(dtype='float64')(c_fc1)

        c_fc2 = Dense(self.concat_fc2_dim, activation='relu',
                      dtype='float64')(c_norm1)
        c_norm2 = BatchNormalization(dtype='float64')(c_fc2)
        c_drop2 = Dropout(0.3)(c_norm2)

        c_fc3 = Dense(self.concat_fc3_dim, activation='relu',
                      dtype='float64')(c_drop2)
        c_norm3 = BatchNormalization(dtype='float64')(c_fc3)

        out = Dense(self.out_dim, activation='tanh', dtype='float64')(c_norm3)
        lamb = Lambda(lambda i: i * self.act_range, dtype='float64')(out)

        model = Model(inputs=[s_inp, e_inp], outputs=[lamb])

        model.compile(optimizer=self.optimizer)

        return model

    # --------------------------------------------------------------------

    def predict(self, states, embeddings):
        return self.model([states, embeddings], training=False)

    # --------------------------------------------------------------------
    def target_predict(self, states, embeddings):
        return self.target_model([states, embeddings], training=False)

    # --------------------------------------------------------------------
    def transferWeights(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = []

        for i in range(len(weights)):
            new_weights.append(
                (self.tau * weights[i]) + ((1.0 - self.tau) * target_weights[i]))

        self.target_model.set_weights(new_weights)

        return

    # --------------------------------------------------------------------
    def saveModel(self, path):
        self.model.save(path + 'actor')
        return

    # --------------------------------------------------------------------
    def loadModel(self, path):
        self.model = load_model(path + 'actor')
        self.target_model = load_model(path + 'actor')
        return

##################################################################


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

        return

    # --------------------------------------------------------------------
    def buildNetwork(self):
        # State input network ---------
        s_inp = Input(shape=(self.state_inp_dim, ))

        s_fc1 = Dense(self.state_fc1_dim, activation='relu',
                      dtype='float64')(s_inp)
        s_norm1 = BatchNormalization(dtype='float64')(s_fc1)

        s_fc2 = Dense(self.state_fc2_dim, activation='relu',
                      dtype='float64')(s_norm1)
        s_norm2 = BatchNormalization(dtype='float64')(s_fc2)
        s_drop2 = Dropout(0.3)(s_norm2)

        # Action input network ---------
        a_inp = Input(shape=(self.action_inp_dim, ))

        a_fc1 = Dense(self.action_fc1_dim, activation='relu',
                      dtype='float64')(a_inp)
        a_norm1 = BatchNormalization(dtype='float64')(a_fc1)
        a_drop1 = Dropout(0.2)(a_norm1)

        # Concatenate the two networks ---
        c_inp = Concatenate(dtype='float64')([s_drop2, a_drop1])

        # Creates the output network
        c_fc1 = Dense(self.conc_fc1_dim, activation='relu',
                      dtype='float64')(c_inp)
        c_norm1 = BatchNormalization(dtype='float64')(c_fc1)

        c_fc2 = Dense(self.conc_fc2_dim, activation='relu',
                      dtype='float64')(c_norm1)
        c_norm2 = BatchNormalization(dtype='float64')(c_fc2)

        out = Dense(self.out_dim, activation='relu', dtype='float64')(c_norm2)

        model = Model(inputs=[s_inp, a_inp], outputs=[out])

        model.compile(optimizer=self.optimizer)

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

        return

    # --------------------------------------------------------------------
    def saveModel(self, path):
        self.model.save(path + 'critic')
        return

    # --------------------------------------------------------------------
    def loadModel(self, path):
        self.model = load_model(path + 'critic')
        self.target_model = load_model(path + 'critic')
        return

##################################################################


class Autoencoder(object):
    def __init__(
        self,
        inputShape,
        lr_init,
        lr_end,
        lr_decay
    ):
        self.lr_init = lr_init
        self.lr = lr_init
        self.inputShape = inputShape
        self.encoder = self.createEncoder()
        self.decoder = self.createDecoder()
        self.autoencoder = self.createAutoencoder()
        self.embedding_model = Model(
            self.encoder.input, Flatten()(self.encoder.output))
        self.lr_end = lr_end
        self.lr_decay = lr_decay

    # ----------------------------------------------------------
    def createEncoder(self):
        encoder = Sequential()

        encoder.add(Conv2D(32, 3, strides=1, padding='same', activation='relu',
                    input_shape=(self.inputShape[0], self.inputShape[1], 3)))
        encoder.add(MaxPooling2D(2, strides=2))

        encoder.add(Conv2D(64, 3, strides=1,
                    padding='same', activation='relu'))
        encoder.add(MaxPooling2D(2, strides=2))
        encoder.add(Dropout(0.2))

        encoder.add(Conv2D(64, 3, strides=1,
                    padding='same', activation='relu'))
        encoder.add(MaxPooling2D(2, strides=2))

        return encoder

    # ----------------------------------------------------------
    def createDecoder(self):
        decoder = Sequential()

        decoder.add(Conv2D(64, 3, strides=1, padding='same',
                    activation='relu', input_shape=self.encoder.output.shape[1:]))
        decoder.add(UpSampling2D(2))

        decoder.add(Conv2D(64, 3, strides=1,
                    padding='same', activation='relu'))
        decoder.add(UpSampling2D(2))
        decoder.add(Dropout(0.2))

        decoder.add(Conv2D(3, 3, strides=1, padding='same', activation='relu'))
        decoder.add(UpSampling2D(2))

        return decoder

    # ----------------------------------------------------------
    def createAutoencoder(self):
        autoencoder = Model(inputs=self.encoder.input,
                            outputs=self.decoder(self.encoder.outputs))

        autoencoder.compile(optimizer=Adam(
            learning_rate=self.lr), loss='mse', metrics=['accuracy'])

        return autoencoder

    # ----------------------------------------------------------
    def feed(self, data):
        data = smart_resize(data, self.inputShape)
        data = self.normalize(data)

        pred = self.autoencoder(np.expand_dims(data, 0), training=False)

        return pred

    # ----------------------------------------------------------
    def train(self, data):
        data = smart_resize(data, self.inputShape)
        data = self.normalize(data)

        self.lr = self.lr * self.lr_decay
        self.lr = min(self.lr, self.lr_end)
        K.set_value(self.autoencoder.optimizer.learning_rate, self.lr)

        history = self.autoencoder.fit(
            data, data, batch_size=5, epochs=2, validation_data=(data, data), verbose=0)
        return history

    # ----------------------------------------------------------
    def getEmbedding(self, data):
        data = smart_resize(data, self.inputShape)
        data = self.normalize(data)

        pred = self.embedding_model(np.expand_dims(data, 0), training=False)

        # return self.normalize(pred)
        return pred

    # ----------------------------------------------------------
    def decode(self, data):
        return self.normalize(self.decoder(data).numpy())

    # ----------------------------------------------------------
    def normalize(self, data):
        amin = np.amin(data)
        amax = np.amax(data)
        data = (data - amin)/(amax - amin)
        return data

    # ----------------------------------------------------------
    def saveModel(self, path):
        self.autoencoder.save(path+'autoencoder')

    # ----------------------------------------------------------
    def loadModel(self, path):
        self.autoencoder = load_model(path+'autoencoder')

    # ----------------------------------------------------------
    @property
    def embeddingDim(self): return self.embedding_model.output.shape[1:]

##################################################################


class DDPGAgent(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        action_min,
        action_max,
        memory_size,
        batch_size,
        gamma,
        a_lr,
        c_lr,
        tau,
        epsilon,
        epsilon_decay,
        epsilon_min,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Creates the Replay Buffer
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        # Creates the autoencoder
        self.autoencoder = Autoencoder((64, 128), 3e-3, 7e-4, 1-5e-6)

        # Creates the actor
        self.actor = Actor(
            state_dim=self.state_dim,
            state_fc1_dim=256,
            state_fc2_dim=128,
            embedding_dim=self.autoencoder.embeddingDim[0],
            embedding_fc1_dim=1024,
            embedding_fc2_dim=256,
            embedding_fc3_dim=64,
            concat_fc1_dim=256,
            concat_fc2_dim=128,
            concat_fc3_dim=32,
            out_dim=self.action_dim,
            act_range=self.action_max,
            lr=self.a_lr,
            tau=self.tau,
        )

        # Creates the critic
        self.critic = Critic(
            state_inp_dim=self.state_dim,
            state_fc1_dim=256,
            state_fc2_dim=128,
            action_inp_dim=self.action_dim,
            action_fc1_dim=32,
            conc_fc1_dim=256,
            conc_fc2_dim=64,
            out_dim=1,
            lr=self.c_lr,
            tau=self.tau,
        )

        # Creates the noise generator
        self.ou_noise = OUActionNoise(mean=np.zeros(action_dim))

    # --------------------------------------------------------------------

    def policy(self, state, embedding, explore=True):
        state = state[np.newaxis, :]
        embedding = embedding[np.newaxis, :]
        action = self.actor.predict(state, embedding)[0]
        # Takes the exploration with the epsilon probability
        if explore and np.random.rand() < self.epsilon:
            action += self.ou_noise()
        action = np.clip(action, a_min=self.action_min, a_max=self.action_max)
        return action

    # --------------------------------------------------------------------
    def learn(self, state, action, reward, next_state, done, embedding, new_embedding):
        self.memory.append(state, action, reward, next_state,
                           done, embedding, new_embedding)
        if self.memory.hasMin:
            self.replay_memory()
        return

    # --------------------------------------------------------------------
    def replay_memory(self):
        # Get sample experiences from the replay buffer
        experiences = self.memory.sample()

        # Get each term of the esxperiences
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        done = np.array([int(exp[4]) for exp in experiences])
        embeddings = np.array([exp[5] for exp in experiences])
        new_embeddings = np.array([exp[6] for exp in experiences])

        # Change the dimensions of the rewards and done arrays
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        # Train the critic
        with GradientTape() as tape:
            # Compute the critic target values
            target_actions = self.actor.target_predict(
                next_states, new_embeddings)
            y = rewards + self.gamma * \
                self.critic.target_predict(
                    next_states, target_actions) * (1 - done)
            # Compute the q_value of each next_state, next_action pair
            critic_value = self.critic.predict(states, actions)
            # Compute the critic loss
            critic_loss = reduce_mean(square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.model.trainable_variables))

        # Train the actor
        with GradientTape() as tape:
            acts = self.actor.predict(states, embeddings)
            critic_grads = self.critic.predict(states, acts)
            # Used -mean as we want to maximize the value given by the critic for our actions
            actor_loss = -reduce_mean(critic_grads)

        actor_grad = tape.gradient(
            actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.model.trainable_variables))

        # Update the model weights
        self.actor.transferWeights()
        self.critic.transferWeights()

        # Decay the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # If its reach the minimum value it stops
        else:
            self.epsilon = self.epsilon_min

        return
    # --------------------------------------------------------------------

    def act(self, env, verbose=False):
        observation = env.reset()
        done = False

        while not done:
            env.render(mode='human')
            time.sleep(0.02)
            embedding = np.squeeze(self.autoencoder.getEmbedding(
                env.render(mode='rgb_array')))
            action = self.policy(observation, embedding, explore=False)
            if verbose:
                print(action)
            new_observation, reward, done, info = env.step(action)
            observation = new_observation

        return

    # --------------------------------------------------------------------
    def train(self, env, envName, num_episodes, verbose, verbose_batch, end_on_complete, complete_num, complete_value, act_after_batch):
        scores_history = []
        steps_history = []
        images = []

        total = 0
        hist = None
        embedding = (np.random.rand(
            self.autoencoder.embeddingDim[0]) - 0.5) * 0.01
        new_embedding = (np.random.rand(
            self.autoencoder.embeddingDim[0]) - 0.5) * 0.01

        # If the complete_num is smaller than 1 ist interpreted as a percentage else its a number of episodes
        if complete_num < 1:
            complete_num = int(
                complete_num*verbose_batch) if int(complete_num*verbose_batch) != 0 else 1

        # Begin the training
        print("BEGIN\n")
        # Number of completed episodes per batch
        complete = 0

        # Iterate on each episode
        for episode in range(num_episodes):
            done = False
            score = 0
            steps = 0
            observation = env.reset()

            while not done:
                action = self.policy(observation, embedding)
                new_observation, reward, done, _ = env.step(action)

                screen = env.render(mode='rgb_array')

                if hist is not None:
                    embedding = new_embedding
                    new_embedding = self.autoencoder.getEmbedding(screen)
                    new_embedding = np.squeeze(new_embedding)
                    loss_mean = np.mean(hist.history['loss'])
                    if loss_mean > 0.004:
                        new_embedding = new_embedding * (1 / (1e4 * loss_mean))

                images.append(screen[150:][:][:])
                self.learn(observation, action, reward,
                           new_observation, done, embedding, new_embedding)

                if(len(images) >= 100):
                    hist = self.autoencoder.train(np.array(images))
                    images = []

                if verbose and total > 100:
                    print(
                        "\r                                                                                                              ", end="")
                    print("\rEpisode: "+str(episode+1)+"\t Step: "+str(steps)+"\tReward: " +
                          str(score)+"\tLoss: "+str(np.mean(hist.history['loss'])), end="")

                observation = new_observation
                score += reward
                steps += 1
                total += 1

            scores_history.append(score)
            steps_history.append(steps)

            # If the score is bigger or equal than the complete score it add one to the completed number
            if(score >= complete_value):
                complete += 1
                # If the flag is true the agent ends the trainig after completing a number of episodes
                if end_on_complete and complete >= complete_num:
                    break

            # These information are printed after each verbose_batch episodes
            if((episode+1) % verbose_batch == 0):
                print("\r                                                                                                          ", end="")
                print("\rEpisodes: ", episode+1, "/", num_episodes, "\n\tTotal reward: ", np.mean(scores_history[-verbose_batch:]), "\n\tLoss: ", np.mean(
                    hist.history['loss']), "\n\tNum. steps: ", np.mean(steps_history[-verbose_batch:]), "\n\tCompleted: ", complete, "\n--------------------------")

                # If the flag is true the agent act and render the episode after each verbose_batch episodes
                if act_after_batch:
                    self.act(env)

                # Set the number of completed episodes on the batch to zero
                complete = 0

        print("\nFINISHED")
        return scores_history, steps_history

    # --------------------------------------------------------------------
    def save(self, path):
        self.actor.saveModel(path)
        self.critic.saveModel(path)
        self.autoencoder.saveModel(path)
        return

    # --------------------------------------------------------------------
    def load(self, a_path, c_path, ae_path):
        self.actor.loadModel(a_path)
        self.critic.loadModel(c_path)
        self.autoencoder.loadModel(ae_path)
        return

##################################################################


name = "BipedalWalkerHardcore-v3"
env = gym.make(name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_min = env.action_space.low
action_max = env.action_space.high

print("State Dimensions: ", state_dim)
print("Actions Dimensions: ", action_dim)
print("Action min: ", action_min)
print("Action max: ", action_max)

memory_size = 10000
batch_size = 128
gamma = 0.9
a_lr = 7e-5
c_lr = 1e-4
tau = 4e-3
epsilon = 1
epsilon_decay = 1-5e-7
epsilon_min = 0.5

agent = DDPGAgent(state_dim, action_dim, action_min, action_max, memory_size,
                  batch_size, gamma, a_lr, c_lr, tau, epsilon, epsilon_decay, epsilon_min)

num_episodes = 3000
verbose = True
verbose_batch = 100
end_on_complete = True
complete_num = 0.5
complete_value = 300
act_after_batch = True

scores, steps = agent.train(env, name, num_episodes, verbose, verbose_batch,
                            end_on_complete, complete_num, complete_value, act_after_batch)

agent.save('networks/11.4_BipedalWalkerHardcore/2022_07_10')
