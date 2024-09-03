import time
import tensorflow as tf
from minisom import MiniSom
import numpy as np
from actor import Actor
from critic import Critic


#####################################################################

# Ornstein-Uhlenbeck Noise


class OUActionNoise(object):
    def __init__(self, mean, sigma=0.4, theta=0.4, dt=7e-2, x0=None):
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

#####################################################################

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
    def append(self, state, action, reward, next_state, done):
        '''
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            done (boolen): True if the next state is a terminal state and False otherwise.
                           Is transformed to integer so tha True = 1, False = 0
            next_state (Numpy array): The next state.           
        '''
        if self.size() == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, next_state, int(done)])

    # --------------------------------------------------------------------------------
    def sample(self, size=None):
        '''
        Returns:
            A list of transition tuples including state, action, reward, terminal, and next_state
        '''
        idxs = self.rand_generator.choice(
            np.arange(len(self.buffer)), size=self.minibatch_size if size is None else size)
        return [self.buffer[idx] for idx in idxs]

    # --------------------------------------------------------------------------------
    def size(self):
        '''
        Returns:
            Number of elements in the buffer
        '''
        return len(self.buffer)

    # --------------------------------------------------------------------------------
    def isMin(self):
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

#####################################################################


class DdpgSomAgent(object):
    def __init__(self, state_dim, action_dim, action_min, action_max, memory_size, batch_size, gamma, a_lr, c_lr, som_lr, tau, epsilon, epsilon_decay, epsilon_min):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.som_lr = som_lr
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Creates the Replay Buffer
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        self.somDimension = 3  # round(math.sqrt(math.log(self.state_dim)))
        self.som = MiniSom(self.somDimension,
                           self.somDimension, self.state_dim, learning_rate=self.som_lr)

        # Creates the actor
        self.actor = Actor(
            inp_dim=self.somDimension*self.somDimension,
            fc1_dim=512,
            fc2_dim=256,
            fc3_dim=64,
            out_dim=self.action_dim,
            act_range=self.action_max,
            lr=self.a_lr,
            tau=self.tau,
        )

        # Creates the critic
        self.critic = Critic(
            state_inp_dim=self.somDimension*self.somDimension,
            state_fc1_dim=128,
            state_fc2_dim=64,
            action_inp_dim=self.action_dim,
            action_fc1_dim=32,
            conc_fc1_dim=128,
            conc_fc2_dim=64,
            out_dim=1,
            lr=self.c_lr,
            tau=self.tau,
        )

        # Creates the noise generator
        self.ou_noise = OUActionNoise(mean=np.zeros(action_dim))

    # --------------------------------------------------------------------
    def policy(self, state, explore=True):
        featMap = self.som.activate(state)
        f_min, f_max = np.amin(featMap), np.amax(featMap)
        featMap = (featMap - f_min) / (f_max - f_min)
        featMap = featMap.reshape(self.somDimension*self.somDimension)
        featMap = featMap[np.newaxis, :]

        action = self.actor.predict(featMap)
        # Takes the exploration with the epsilon probability

        if explore and np.random.rand() < self.epsilon:
            action += self.ou_noise()

        action = np.clip(action, a_min=self.action_min, a_max=self.action_max)
        return action[0], featMap

    # --------------------------------------------------------------------
    def getMaps(self, states):
        maps = []
        for state in states:
            action, map = self.policy(state)
            maps.append(map[0])

        return np.array(maps, dtype=np.float32)

    # --------------------------------------------------------------------
    def learn(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

        if self.memory.isMin():
            self.train_som()
            self.replay_memory()

    # --------------------------------------------------------------------
    def train_som(self):
        # Get sample experiences from the replay buffer
        experiences = self.memory.sample()

        # Get each term of the experiences
        states = np.array([exp[0] for exp in experiences])

        self.som.train(states, num_iteration=len(states))

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

        # Change the dimensions of the rewards and done arrays
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        maps = self.getMaps(states)
        next_maps = self.getMaps(next_states)

        # Train the critic
        with tf.GradientTape() as tape:
            # Compute the critic target values
            target_actions = self.actor.target_predict(next_maps)
            y = rewards + self.gamma * \
                self.critic.target_predict(
                    next_maps, target_actions) * (1 - done)
            # Compute the q_value of each next_state, next_action pair
            critic_value = self.critic.predict(maps, actions)
            # Compute the critic loss
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.model.trainable_variables))

        # Train the actor
        with tf.GradientTape() as tape:
            acts = self.actor.predict(maps)
            critic_grads = self.critic.predict(maps, acts)
            # Used -mean as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_grads)

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

    # --------------------------------------------------------------------
    def act(self, env):
        # Reset the envirorment
        observation = env.reset()
        done = False

        while not done:
            # env.render()
            time.sleep(0.02)
            action, map = self.policy(observation, explore=False)
            new_observation, reward, done, info = env.step(action)
            observation = new_observation

        # env.close()

    # --------------------------------------------------------------------
    def train(self, env, num_episodes, verbose, verbose_batch, end_on_complete, complete_num, complete_value, act_after_batch):
        scores_history = []
        steps_history = []

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
                action, map = self.policy(observation)

                if verbose:
                    print(
                        "\r                                                                                                     ", end="")
                    print("\rEpisode: "+str(episode+1)+"\t Step: " +
                          str(steps)+"\tReward: "+str(score), end="")

                new_observation, reward, done, _ = env.step(action)
                self.learn(observation, action, reward, new_observation, done)
                observation = new_observation
                score += reward
                steps += 1

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
                print("\rEpisodes: ", episode+1, "/", num_episodes, "\n\tTotal reward: ", np.mean(scores_history[-verbose_batch:]), "\n\tNum. steps: ", np.mean(
                    steps_history[-verbose_batch:]), "\n\tCompleted: ", complete, "\n--------------------------")

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

    # --------------------------------------------------------------------
    def load(self, a_path, c_path):
        self.actor.loadModel(a_path)
        self.critic.loadModel(c_path)
