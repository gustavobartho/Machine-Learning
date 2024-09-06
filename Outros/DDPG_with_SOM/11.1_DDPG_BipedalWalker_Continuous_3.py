import gym
import tensorflow as tf, matplotlib.pyplot as plt, numpy as np

from tensorflow.keras.layers import Input, Concatenate, Lambda, Layer, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.initializers import RandomUniform

global_seed = 42
tf.random.set_seed(global_seed)
np.random.seed(global_seed)

########################################

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
    

##################################################

class PrioritizedReplayBuffer:

    def __init__(self, capacity, batch_size, alpha=0.3, beta=0.2, beta_increment=0.003):
        # Initialize buffer parameters
        self.capacity = capacity  # Maximum number of experiences to store
        self.batch_size = batch_size  # Number of experiences to sample in each batch
        self.alpha = alpha  # Exponent for prioritization (0 = uniform, 1 = full prioritization)
        self.beta = beta  # Initial importance sampling weight
        self.beta_increment = beta_increment  # Increment for beta over time
        
        # Initialize buffer and priorities
        self.buffer = np.zeros((capacity, 5, ), dtype=object)  # Buffer to store experiences
        self.priorities = np.zeros(capacity, dtype=np.float32) + 1e-6  # Priorities for each experience
        self.position = 0  # Current position in the buffer
        self.size = 0  # Current size of the buffer


    def append(self, state, action, reward, next_state, done):
        # Get the maximum priority in the buffer (for new experiences)
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        # Store the new experience in the buffer
        self.buffer[self.position] = [state, action, reward, next_state, done]
        
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
    
    
#################################################

class VAE:
    def __init__(self, input_dim, latent_dim, encoder_dims, lr):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.lr = lr
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.encoder_optimizer = Adam(learning_rate=lr)
        self.decoder_optimizer = Adam(learning_rate=lr)


    def build_encoder(self):
        inputs = Input(shape=(self.input_dim,))
        x = inputs

        for dim in self.encoder_dims:
            x = Dense(dim, activation='relu')(x)

        mean = Dense(self.latent_dim)(x)
        logvar = Dense(self.latent_dim)(x)
        z = Lambda(lambda x: self.reparameterize(x[0], x[1]))([mean, logvar])
        return Model(inputs, [z, mean, logvar])


    def build_decoder(self):
        inputs = Input(shape=(self.latent_dim,))
        x = inputs

        for dim in reversed(self.encoder_dims):
            x = Dense(dim, activation='relu')(x)

        outputs = Dense(self.input_dim)(x)
        return Model(inputs, outputs)
    

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean


    @tf.function
    def encode(self, x):
        return self.encoder(x)
    

    @tf.function
    def decode(self, z):
        return self.decoder(z)


    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, mean, logvar = self.encode(x)
            x_reconstructed = self.decode(z)
            reconstruction_loss = tf.reduce_mean(tf.square(x - x_reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(gradients[:len(self.encoder.trainable_variables)], self.encoder.trainable_variables))
        self.decoder_optimizer.apply_gradients(zip(gradients[len(self.encoder.trainable_variables):], self.decoder.trainable_variables))

        return total_loss, reconstruction_loss, kl_loss
    


#################################################


class MultiHeadActor(object):
    
    def __init__(self, inp_dim, heads_nets_dims, kernel_probs_net_dims, out_dim, n_kernels, act_range, lr, tau):
        # Initialize the Multi-Head Actor with given parameters
        self.inp_dim = inp_dim  # Dimension of the state input
        self.heads_nets_dims = heads_nets_dims
        self.kernel_probs_net_dims = kernel_probs_net_dims
        self.out_dim = out_dim  # Dimension of each output head
        self.n_kernels = n_kernels  # Number of output heads (kernels)
        self.act_range = act_range  # Action range for scaling the output
        self.tau = tau  # Soft update parameter for target network

        self.vae = VAE(input_dim=inp_dim, latent_dim=8, encoder_dims=[64, 32], lr=1e-3)

        self.model = self.buildNetwork()
        self.target_model = self.buildNetwork()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(learning_rate=lr)


    def buildNetwork(self):
        inp = Input(shape=(self.inp_dim,))
        z, _, _ = self.vae.encoder(inp)
        
        # Kernel probabilities network
        kernel_probs = z
        for dim in self.kernel_probs_net_dims:
            kernel_probs = Dense(dim, activation='relu')(kernel_probs)

        kernel_probs = Dense(self.n_kernels, activation='softmax')(kernel_probs)


        # Specialist networks
        specialist_outputs = []
        for _ in range(self.n_kernels):
            specialist = inp
            for dim in self.heads_nets_dims:
                specialist = Dense(dim, activation='relu')(specialist)
                specialist = Dropout(0.2)(specialist)

            specialist = Dense(tf.round(self.out_dim/self.n_kernels))(specialist)
            specialist = Dropout(0.2)(specialist)
            specialist_outputs.append(specialist)

        # Gate and combine specialist outputs
        gated_outputs = [
            Lambda(lambda x: x[0] * x[1])([kernel_probs[:, i:i+1], specialist]) 
            for i, specialist in enumerate(specialist_outputs)
        ]

        combined_output = Concatenate(dtype=tf.float32)(gated_outputs)
        output = Dense(4*self.out_dim*self.n_kernels, activation='relu')(combined_output)

        output = NoisyDense(self.out_dim, activation='relu')(output)
        output = NoisyDense(self.out_dim, activation='relu')(output)
        output = NoisyDense(self.out_dim, activation='tanh')(output)
        output = Lambda(lambda x: x * self.act_range)(output)

        return Model(inputs=inp, outputs=output)


    @tf.function
    def predict(self, states):
        return self.model(states)


    @tf.function
    def target_predict(self, states):
        return self.target_model(states)


    @tf.function
    def transferWeights(self):
        for a, b in zip(self.target_model.variables, self.model.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)


########################################


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
        self.optimizer = Adam(learning_rate=lr)
        
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
            kernel_initializer=RandomUniform(-f1, f1), 
            bias_initializer=RandomUniform(-f1, f1), dtype=tf.float32
        )(s_inp)
        s_norm1 = BatchNormalization(dtype=tf.float32)(s_fc1)
        
        # Action input branch
        a_inp = Input(shape=(self.action_inp_dim, ))
        f1 = 1 / np.sqrt(self.action_fc1_dim)
        a_fc1 = Dense(
            self.action_fc1_dim, activation='relu', 
            kernel_initializer=RandomUniform(-f1, f1), 
            bias_initializer=RandomUniform(-f1, f1), dtype=tf.float32
        )(a_inp)
        a_norm1 = BatchNormalization(dtype=tf.float32)(a_fc1)
        
        # Concatenate state and action branches
        c_inp = Concatenate(dtype=tf.float32)([s_norm1, a_norm1])
        
        # Fully connected layers after concatenation
        f1 = 1 / np.sqrt(self.conc_fc1_dim)
        c_fc1 = Dense(
            self.conc_fc1_dim, activation='relu', 
            kernel_initializer=RandomUniform(-f1, f1), 
            bias_initializer=RandomUniform(-f1, f1), dtype=tf.float32
        )(c_inp)
        c_norm1 = BatchNormalization(dtype=tf.float32)(c_fc1)

        f2 = 1 / np.sqrt(self.conc_fc2_dim)
        c_fc2 = Dense(
            self.conc_fc2_dim, activation='relu', 
            kernel_initializer=RandomUniform(-f2, f2), 
            bias_initializer=RandomUniform(-f2, f2), dtype=tf.float32
        )(c_norm1)
        c_norm2 = BatchNormalization(dtype=tf.float32)(c_fc2)
        
        # Output layer
        f3 = 0.003
        out = Dense(
            self.out_dim, activation='linear', 
            kernel_initializer=RandomUniform(-f3, f3), 
            bias_initializer=RandomUniform(-f3, f3), dtype=tf.float32
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



########################################

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

        # Initialize actor network
        self.actor = MultiHeadActor(
            inp_dim=self.state_dim, 
            heads_nets_dims=[512, 256, 128, 64],
            kernel_probs_net_dims = [16, 16],
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
        #self.create_plot()


    def create_plot(self):
        # Create a figure for SOM activation visualization
        self.fig = plt.figure()
        self.som_act_plot = self.fig.add_subplot(211)
        self.som_act_plot.title.set_text('SOM Activation')

        self.kernel_plot = self.fig.add_subplot(212)
        self.kernel_plot.title.set_text('Kernels centers')
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
        action = self.actor.predict(tf.expand_dims(state, 0))
        action = tf.clip_by_value(action, self.action_min, self.action_max)
        return action[0]


    @tf.function
    def update_nets(self, weights, states, actions, rewards, next_states, dones):
        weights = tf.cast(weights, dtype=tf.float32)
        # Update critic network
        with tf.GradientTape() as tape:
            target_actions = self.actor.target_predict(next_states)
            y = tf.cast(rewards + self.gamma * self.critic.target_predict(next_states, target_actions) * (1 - dones), dtype=tf.float32)
            critic_value = tf.cast(self.critic.predict(states, actions), dtype=tf.float32)
            critic_loss = tf.reduce_mean(weights * tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.model.trainable_variables))
        
        # Update actor network
        with tf.GradientTape() as tape:
            actions = self.actor.predict(states)
            critic_grads = self.critic.predict(states, actions)
            actor_loss = -tf.math.reduce_mean(critic_grads)
        
        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))

        self.actor.transferWeights()
        self.critic.transferWeights()

        return y, critic_value


    def learn(self, state, action, reward, next_state, done):
        # Add experience to replay buffer and perform learning if buffer is sufficiently filled
        self.memory.append(state, action, reward, next_state, done)
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

        self.actor.vae.train_step(tf.concat([states, next_states], axis=0))
        y, critic_value = self.update_nets(weights, states, actions, rewards, next_states, dones)

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
            action = self.policy(observation)
            new_observation, _, done, _, _ = env2.step(action.numpy())
            observation = new_observation
            step += 1
            done = done or (step > self.max_steps)
        
        env2.close()
        return


    def train(
        self, env, num_episodes, verbose, verbose_num, end_on_complete, 
        complete_num, complete_value, act_after_batch
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
                action = self.policy(observation)
                
                if verbose:
                    print("\r                                                          ", end="")
                    print(f"\rEpisode: {str(episode+1)} \tStep: {str(steps)} \tReward: {str(score)}", end="")
                
                new_observation, reward, done, _, _ = env.step(action.numpy())
                
                if steps > self.max_steps:
                    reward = -50
                    done = True

                self.learn(observation, action.numpy(), reward, new_observation, done)
                observation = new_observation
                score += reward
                steps += 1

            scores_history.append(score)
            steps_history.append(steps)
            
            if score >= complete_value:
                complete += 1
                if end_on_complete and complete >= complete_num: break
            
            if (episode+1) % verbose_num == 0:
                print("\r                                                 ", end="")
                print(f'''\rEpisodes: {episode+1}/{num_episodes}\n\tTotal reward: {np.mean(scores_history[-verbose_num:])} +- {np.std(scores_history[-verbose_num:])}\n\tNum. steps: {np.mean(steps_history[-verbose_num:])} +- {np.std(steps_history[-verbose_num:])}\n\tCompleted: {complete}\n--------------------------''')
                
                if act_after_batch: self.act()
                complete = 0

        print("\nFINISHED")
        
        return scores_history, steps_history


########################################


name = "BipedalWalker-v3"
env = gym.make(name, hardcore=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_min = env.action_space.low
action_max = env.action_space.high

memory_size = 1000000
batch_size = 128
gamma = 0.99
a_lr = 1e-4
c_lr = 1e-3
tau = 5e-3
max_steps = 1000
n_kernels = 3

agent = DDPGAgent(
    state_dim, action_dim, action_min, action_max, 
    memory_size, batch_size, gamma, a_lr, c_lr, tau, 
    max_steps, name, n_kernels,
)


num_episodes = 3000
verbose = True
verbose_num = 5
end_on_complete = True
complete_num = 2
complete_value = 300
act_after_batch = True

agent.train(
    env, num_episodes, verbose, 
    verbose_num, end_on_complete,  
    complete_num, complete_value, 
    act_after_batch
)