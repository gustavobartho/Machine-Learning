import gym, warnings
import tensorflow as tf, matplotlib.pyplot as plt, numpy as np

from tensorflow.keras.layers import Input, Concatenate, Lambda, Layer, Reshape, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from minisom import MiniSom
from sklearn.cluster import KMeans

global_seed = 42
tf.random.set_seed(global_seed)
np.random.seed(global_seed)
warnings.filterwarnings("ignore")


# NoisyDense Layer: A custom layer that adds parametric noise to the weights and biases
# This can help with exploration in reinforcement learning tasks
class NoisyDense(Layer):
    
    def __init__(self, units, activation=None):
        super(NoisyDense, self).__init__()
        self.units = units  # Number of output units
        self.activation = tf.keras.activations.get(activation)  # Activation function


    def build(self, input_shape):
        # Initialize learnable parameters for mean and standard deviation of weights and biases
        self.w_mu = self.add_weight("w_mu", shape=(input_shape[-1], self.units))
        self.w_sigma = self.add_weight("w_sigma", shape=(input_shape[-1], self.units))
        self.b_mu = self.add_weight("b_mu", shape=(self.units,))
        self.b_sigma = self.add_weight("b_sigma", shape=(self.units,))


    def call(self, inputs):
        # Generate random noise for weights and biases
        w_epsilon = tf.random.normal(self.w_mu.shape)
        b_epsilon = tf.random.normal(self.b_mu.shape)
        
        # Combine mean and noise to create noisy weights and biases
        w = self.w_mu + self.w_sigma * w_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon
        
        # Perform the dense layer operation
        output = tf.matmul(inputs, w) + b
        
        # Apply activation function if specified
        return self.activation(output) if self.activation else output
    

class PrioritizedReplayBuffer:

    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        # Initialize buffer parameters
        self.capacity = capacity  # Maximum number of experiences to store
        self.batch_size = batch_size  # Number of experiences to sample in each batch
        self.alpha = alpha  # Exponent for prioritization (0 = uniform, 1 = full prioritization)
        self.beta = beta  # Initial importance sampling weight
        self.beta_increment = beta_increment  # Increment for beta over time
        
        # Initialize buffer and priorities
        self.buffer = np.zeros((capacity, 6, ), dtype=object)  # Buffer to store experiences
        self.priorities = np.zeros(capacity, dtype=np.float32) + 1e-6  # Priorities for each experience
        self.position = 0  # Current position in the buffer
        self.size = 0  # Current size of the buffer


    def append(self, state, action, reward, next_state, done, kernel_probs):
        # Get the maximum priority in the buffer (for new experiences)
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        # Store the new experience in the buffer
        self.buffer[self.position] = [state, action, reward, next_state, done, kernel_probs]
        
        # Assign max priority to the new experience
        self.priorities[self.position] = max_priority
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self):
        # Check if there are enough samples in the buffer
        if self.size < self.batch_size:
            return [], [], []

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)

        # Sample indices based on priorities
        indices = np.random.choice(self.size, self.batch_size, p=probabilities, replace=False)
        
        # Get the sampled experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** -self.beta
        weights /= np.max(weights)
        
        # Increase beta for future sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights


    def update_priorities(self, indices, priorities):
        # Update priorities for the sampled experiences
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Add small constant to avoid zero priority


    def isMin(self):
        # Check if the buffer has enough samples for a full batch
        return self.size >= self.batch_size
    

class InstinctiveNetwork:
    
    def __init__(self, som_dim: int, input_dim: int, n_kernels: int, som_kwargs: dict):
        # Initialize SOM dimensions
        self.som_dim = som_dim
        self.som_dims = (som_dim, som_dim)
        
        # Set input dimension
        self.input_dim = input_dim
        
        # Create SOM instance
        self.som = MiniSom(*self.som_dims, self.input_dim, **som_kwargs)
        
        # Initialize other attributes
        self.n_kernels = n_kernels
        self.kmeans = KMeans(n_clusters=self.n_kernels, random_state=42)
        self.nn_trained = False
        self.kernel_centers = None


    @tf.function
    def get_output(self, data):
        # Activate SOM
        som_act = self._tf_activate(data)
        
        # Normalize SOM activation
        som_act = 1 - ((som_act - tf.reduce_min(som_act)) / (tf.reduce_max(som_act) - tf.reduce_min(som_act)))

        # Train neural network if not already trained
        if not self.nn_trained:
            self.train_nn()

        # Calculate distances to kernel centers
        distances = tf.norm(tf.expand_dims(self.kernel_centers, 0) - data, axis=-1)

        # Convert distances to probabilities
        kernel_probs = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
        kernel_probs /= tf.reduce_sum(kernel_probs, axis=-1, keepdims=True)  # Normalize to sum to 1

        return som_act, kernel_probs


    @tf.function
    def _tf_activate(self, x):
        # Expand input dimensions
        x = tf.expand_dims(x, 0)
        # Convert SOM weights to TensorFlow constant
        w = tf.constant(self.som._weights, dtype=tf.float32)
        # Calculate Euclidean distance between input and SOM weights
        return tf.norm(tf.subtract(x, w), axis=-1)


    def train_som(self, data):
        # Train SOM using the provided data
        self.som.train(data, 5, use_epochs=True)
        
        # Train neural network after SOM training
        self.train_nn()


    def train_nn(self):
        # Reshape SOM weights for clustering
        som_weights = self.som.get_weights().reshape(-1, self.input_dim)
        
        # Perform K-means clustering on SOM weights
        self.kmeans.fit(som_weights)
        
        # Set kernel centers to K-means cluster centers
        self.kernel_centers = tf.constant(self.kmeans.cluster_centers_, dtype=tf.float32)
        
        # Mark neural network as trained
        self.nn_trained = True

class MultiHeadActor(object):
    
    def __init__(self, s_inp_dim, s_fc1_dim, fc2_dim, fc3_dim, out_dim, n_kernels, act_range, lr, tau):
        # Initialize the Multi-Head Actor with given parameters
        self.s_inp_dim = s_inp_dim  # Dimension of the state input
        self.s_fc1_dim = s_fc1_dim  # Dimension of the first fully connected layer
        self.fc2_dim = fc2_dim  # Dimension of the second fully connected layer
        self.fc3_dim = fc3_dim  # Dimension of the third fully connected layer
        self.out_dim = out_dim  # Dimension of each output head
        self.n_kernels = n_kernels  # Number of output heads (kernels)
        self.act_range = act_range  # Action range for scaling the output
        self.tau = tau  # Soft update parameter for target network
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=0.99
        )
        self.optimizer = Adam(learning_rate=self.lr_schedule)
        self.model = self.buildNetwork()  # Build the main network
        self.target_model = self.buildNetwork()  # Build the target network
        self.target_model.set_weights(self.model.get_weights())  # Initialize target network weights


    def buildNetwork(self):
        # Build the neural network architecture
        inp = Input(shape=(self.s_inp_dim,))  # Input layer
        
        # First fully connected layer
        f1 = 1 / np.sqrt(self.s_fc1_dim)  # Scaling factor for weight initialization
        fc1 = Dense(self.s_fc1_dim, activation='relu', 
                    kernel_initializer=tf.random_uniform_initializer(-f1, f1), 
                    bias_initializer=tf.random_uniform_initializer(-f1, f1), 
                    dtype='float64')(inp)
        norm1 = BatchNormalization(dtype='float64')(fc1)  # Batch normalization
        
        # Second fully connected layer
        fc2 = NoisyDense(self.fc2_dim, activation='relu')(norm1)
        norm2 = BatchNormalization(dtype='float64')(fc2)

        # Third fully connected layer
        f3 = 1 / np.sqrt(self.fc3_dim)
        fc3 = Dense(self.fc3_dim, activation='relu', 
                    kernel_initializer=tf.random_uniform_initializer(-f3, f3), 
                    bias_initializer=tf.random_uniform_initializer(-f3, f3), 
                    dtype='float64')(norm2)
        norm3 = BatchNormalization(dtype='float64')(fc3)

        # Multiple output heads, one for each kernel
        outputs = [NoisyDense(self.out_dim, activation='tanh')(norm3) for _ in range(self.n_kernels)]
        outputs = Concatenate(axis=-1)(outputs)  # Concatenate all outputs
        outputs = Reshape((self.n_kernels, self.out_dim))(outputs)  # Reshape to (n_kernels, out_dim)
        outputs = Lambda(lambda x: x * self.act_range)(outputs)  # Scale outputs to action range
        
        model = Model(inputs=inp, outputs=outputs)  # Create the model
        return model


    def predict(self, states):
        # Predict actions for given states using the main network
        return self.model(states)


    def target_predict(self, states):
        # Predict actions for given states using the target network
        return self.target_model(states)


    @tf.function
    def transferWeights(self):
        # Soft update of target network weights
        for target_weight, weight in zip(self.target_model.weights, self.model.weights):
            target_weight.assign(self.tau * weight + (1 - self.tau) * target_weight)


class Critic(object):
    
    def __init__(self, state_inp_dim, state_fc1_dim, action_inp_dim, action_fc1_dim, conc_fc1_dim, conc_fc2_dim, out_dim, lr, tau):
        # Initialize the Critic network dimensions and parameters
        self.state_inp_dim = state_inp_dim
        self.state_fc1_dim = state_fc1_dim
        self.action_inp_dim = action_inp_dim
        self.action_fc1_dim = action_fc1_dim
        self.conc_fc2_dim = conc_fc2_dim
        self.conc_fc1_dim = conc_fc1_dim
        self.out_dim = out_dim
        
        # Set up learning rate schedule
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=0.99
        )
        
        # Initialize optimizer
        self.optimizer = Adam(learning_rate=self.lr_schedule)
        
        # Parameter for soft updates of target network
        self.tau = tau
        
        # Build the main critic network
        self.model = self.buildNetwork()
        
        # Build the target critic network
        self.target_model = self.buildNetwork()
        
        # Initialize target network weights to match main network
        self.target_model.set_weights(self.model.get_weights())


    def buildNetwork(self):
        # State input branch
        s_inp = Input(shape=(self.state_inp_dim, ))
        f1 = 1 / np.sqrt(self.state_fc1_dim)
        s_fc1 = Dense(
            self.state_fc1_dim, activation='relu', 
            kernel_initializer=tf.random_uniform_initializer(-f1, f1), 
            bias_initializer=tf.random_uniform_initializer(-f1, f1), dtype='float64'
        )(s_inp)
        s_norm1 = BatchNormalization(dtype='float64')(s_fc1)
        
        # Action input branch
        a_inp = Input(shape=(self.action_inp_dim, ))
        f1 = 1 / np.sqrt(self.action_fc1_dim)
        a_fc1 = Dense(
            self.action_fc1_dim, activation='relu', 
            kernel_initializer=tf.random_uniform_initializer(-f1, f1), 
            bias_initializer=tf.random_uniform_initializer(-f1, f1), dtype='float64'
        )(a_inp)
        a_norm1 = BatchNormalization(dtype='float64')(a_fc1)
        
        # Concatenate state and action branches
        c_inp = Concatenate(dtype='float64')([s_norm1, a_norm1])
        
        # Fully connected layers after concatenation
        f1 = 1 / np.sqrt(self.conc_fc1_dim)
        c_fc1 = Dense(
            self.conc_fc1_dim, activation='relu', 
            kernel_initializer=tf.random_uniform_initializer(-f1, f1), 
            bias_initializer=tf.random_uniform_initializer(-f1, f1), dtype='float64'
        )(c_inp)
        c_norm1 = BatchNormalization(dtype='float64')(c_fc1)

        f2 = 1 / np.sqrt(self.conc_fc2_dim)
        c_fc2 = Dense(
            self.conc_fc2_dim, activation='relu', 
            kernel_initializer=tf.random_uniform_initializer(-f2, f2), 
            bias_initializer=tf.random_uniform_initializer(-f2, f2), dtype='float64'
        )(c_norm1)
        c_norm2 = BatchNormalization(dtype='float64')(c_fc2)
        
        # Output layer
        f3 = 0.003
        out = Dense(
            self.out_dim, activation='linear', 
            kernel_initializer=tf.random_uniform_initializer(-f3, f3), 
            bias_initializer=tf.random_uniform_initializer(-f3, f3), dtype='float64'
        )(c_norm2)
        
        # Create and return the model
        model = Model(inputs=[s_inp, a_inp], outputs=[out])
        return model


    @tf.function
    def predict(self, states, actions):
        # Make predictions using the main network
        return tf.cast(self.model([states, actions], training=False), tf.float32)


    @tf.function
    def target_predict(self, states, actions):
        # Make predictions using the target network
        return tf.cast(self.target_model([states, actions], training=False), tf.float32)


    @tf.function
    def transferWeights(self):
        # Perform soft update of target network weights
        for target_weight, weight in zip(self.target_model.weights, self.model.weights):
            target_weight.assign(self.tau * weight + (1 - self.tau) * target_weight)
        
    def saveModel(self, path):
        # Save the weights of the main network
        self.model.save_weights(path + '_critic.h5')
    
    def loadModel(self, path):
        # Load the weights into the main network
        self.model.load_weights(path)


class DDPGAgent(object):
    def __init__(
        self, state_dim, action_dim, action_min, action_max, 
        memory_size, batch_size, gamma, a_lr, c_lr, tau, max_steps, 
        env_name, n_kernels
    ):
        # Initialize agent parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.a_lr = a_lr  # Actor learning rate
        self.c_lr = c_lr  # Critic learning rate
        self.tau = tau  # Soft update parameter
        self.max_steps = max_steps
        self.env_name = env_name
        self.n_kernels = n_kernels

        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(memory_size, batch_size)

        # Initialize SOM-based instinctive network
        som_dim = (20, 20)
        self.inst_net = InstinctiveNetwork(
            som_dim=som_dim[0], 
            input_dim=self.state_dim,
            n_kernels=self.n_kernels,
            som_kwargs = {
                'sigma': 0.7,
                'learning_rate': 0.01,
                'neighborhood_function': 'gaussian',
                'random_seed': 42,
                'decay_function': lambda x, y, z: x,
            },
        )

        # Initialize actor network
        self.actor = MultiHeadActor(
            s_inp_dim=self.state_dim, 
            s_fc1_dim=512,
            fc2_dim=256, 
            fc3_dim=64,
            out_dim=self.action_dim,
            n_kernels=self.n_kernels,
            act_range=self.action_max, 
            lr=self.a_lr, 
            tau=self.tau,
        )

        # Initialize critic network
        self.critic = Critic(
            state_inp_dim=self.state_dim, 
            state_fc1_dim=512, 
            action_inp_dim=self.action_dim, 
            action_fc1_dim=128,
            conc_fc1_dim=256, 
            conc_fc2_dim=64,
            out_dim=1,
            lr=self.c_lr, 
            tau=self.tau,
        )

        # Create plot for visualization
        self.create_plot()


    def create_plot(self):
        # Create a figure for SOM activation visualization
        self.fig = plt.figure()
        self.som_act_plot = self.fig.add_subplot(111)
        self.som_act_plot.title.set_text('SOM Activation')
        return


    def update_plots(self, som_act):
        # Update the SOM activation plot
        self.som_act_plot.imshow(som_act)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return


    @tf.function
    def policy(self, state):
        # Generate action based on current policy
        state = tf.expand_dims(state, 0)
        som_act, kernel_probs = self.inst_net.get_output(state)
        actions = self.actor.predict(state)  # Shape: [1, n_kernels, action_dim]
        action = tf.reduce_sum(actions * tf.expand_dims(kernel_probs, -1), axis=1)[0]
        action = tf.clip_by_value(action, self.action_min, self.action_max)
        return action, som_act, kernel_probs[0]


    @tf.function
    def update_nets(self, weights, states, actions, rewards, next_states, dones, kernel_probs):
        # Update critic network
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_predict(next_states)
            kernel_probs_expanded = tf.expand_dims(kernel_probs, axis=-1)
            weighted_target_actions = tf.reduce_sum(target_actions * kernel_probs_expanded, axis=1)
            
            y = rewards + self.gamma * self.critic.target_predict(next_states, weighted_target_actions) * (1 - dones)
            critic_value = self.critic.predict(states, actions)
            critic_loss = tf.reduce_mean(weights * tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.model.trainable_variables))
        
        # Update actor network
        with tf.GradientTape() as tape:
            actions = self.actor.predict(states)
            weighted_actions = tf.reduce_sum(actions * kernel_probs_expanded, axis=1)
            critic_value = self.critic.predict(states, weighted_actions)
            actor_loss = -tf.reduce_mean(weights * critic_value)
            
        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))
        
        # Soft update target networks
        self.actor.transferWeights()
        self.critic.transferWeights()

        return y, critic_value


    def learn(self, state, action, reward, next_state, done, kernel_probs):
        # Add experience to replay buffer and perform learning if buffer is sufficiently filled
        self.memory.append(state, action, reward, next_state, done, kernel_probs)
        self.replay_memory()


    def replay_memory(self):
        if not self.memory.isMin(): return  # Not enough samples in the buffer
        
        # Sample from replay buffer and perform learning update
        experiences, indices, weights = self.memory.sample()
        
        states = tf.convert_to_tensor([exp[0] for exp in experiences], dtype=tf.float32)
        actions = tf.convert_to_tensor([exp[1] for exp in experiences], dtype=tf.float32)
        rewards = tf.convert_to_tensor([exp[2] for exp in experiences], dtype=tf.float32)
        next_states = tf.convert_to_tensor([exp[3] for exp in experiences], dtype=tf.float32)
        dones = tf.convert_to_tensor([exp[4] for exp in experiences], dtype=tf.float32)
        kernel_probs = tf.convert_to_tensor([exp[5] for exp in experiences], dtype=tf.float32)

        y, critic_value = self.update_nets(weights, states, actions, rewards, next_states, dones, kernel_probs)

        # Update priorities in the replay buffer
        td_errors = tf.abs(y - critic_value)
        self.memory.update_priorities(indices, td_errors.numpy().flatten())
        return


    def act(self):
        # Perform a single episode in the environment using the current policy
        env2 = gym.make(self.env_name, hardcore=True, render_mode='human')
        observation, _ = env2.reset()
        done = False
        step = 0
        
        while not done:
            env2.render()
            action, _, _ = self.policy(observation)
            new_observation, _, done, _, _ = env2.step(action.numpy())
            observation = new_observation
            step += 1
            done = done or (step > self.max_steps)
        
        env2.close()
        return


    def train(
        self, env, num_episodes, verbose, verbose_num, end_on_complete, 
        complete_num, complete_value, act_after_batch, plot_act
    ):
        # Main training loop
        scores_history = []
        steps_history = []

        print("BEGIN\n")
        complete = 0
        
        for episode in range(num_episodes):
            done = False
            score = 0
            steps = 0
            observation, _ = env.reset()
            
            while not done:
                action, som_act, kernel_probs = self.policy(observation)
                
                if verbose: print(f"\rEpisode: {episode+1} \tStep: {steps} \tReward: {score}", end="")
                
                new_observation, reward, done, _, _ = env.step(action.numpy())
                
                if steps > self.max_steps:
                    reward = -50
                    done = True

                self.learn(observation, action.numpy(), reward, new_observation, done, kernel_probs)
                observation = new_observation
                score += reward
                steps += 1

                if plot_act: self.update_plots(som_act)

            scores_history.append(score)
            steps_history.append(steps)
            
            if score >= complete_value:
                complete += 1
                if end_on_complete and complete >= complete_num: break
            
            if (episode+1) % verbose_num == 0:
                print(f'''\rEpisodes: {episode+1}/{num_episodes}\n\tTotal reward: {np.mean(scores_history[-verbose_num:])} +- {np.std(scores_history[-verbose_num:])}\n\tNum. steps: {np.mean(steps_history[-verbose_num:])} +- {np.std(steps_history[-verbose_num:])}\n\tCompleted: {complete}\n--------------------------''')
                
                if act_after_batch: self.act()
                complete = 0

        print("\nFINISHED")
        
        return scores_history, steps_history
    

name = "BipedalWalker-v3"
env = gym.make(name, hardcore=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_min = env.action_space.low
action_max = env.action_space.high

memory_size = 1000000
batch_size = 256
gamma = 0.99
a_lr = 3e-4
c_lr = 8e-4
tau = 1e-3
max_steps = 1000
n_kernels = 10

agent = DDPGAgent(
    state_dim, action_dim, action_min, action_max, 
    memory_size, batch_size, gamma, a_lr, c_lr, tau, 
    max_steps, name, n_kernels,
)

num_episodes = 3000
verbose = True
verbose_num = 50
end_on_complete = True
complete_num = 2
complete_value = 300
act_after_batch = True

agent.train(
    env, num_episodes, verbose, 
    verbose_num, end_on_complete, 
    complete_num, complete_value, 
    act_after_batch, plot_act=False
)
