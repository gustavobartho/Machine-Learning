import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Concatenate, Lambda, Layer, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.initializers import RandomUniform

global_seed = 42
tf.random.set_seed(global_seed)
np.random.seed(global_seed)

###############################################

class PrioritizedReplayBuffer:

    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_increment=1e-3):
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
            self.priorities[idx] = priority + 1e-6  # Add small constant to avoid zero priority


    def isMin(self):
        # Check if the buffer has enough samples for a full batch
        return self.size >= self.batch_size

###############################################

class TD3Actor(object):
    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.noise_std = 0.1
        self.noise_clip = 0.5

        self.model = self.buildNetwork()
        self.target_model = self.buildNetwork()
        self.target_model.set_weights(self.model.get_weights())

        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=100000,
            decay_rate=0.99
        )
        self.optimizer = Adam(learning_rate=self.lr_schedule)


    def buildNetwork(self):
        inp = Input(shape=(self.inp_dim,))
        x = Dense(1024, activation='relu')(inp)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(self.out_dim, activation='tanh')(x)
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
            

    def add_noise(self, actions):
        noise = np.random.normal(0, self.noise_std, size=actions.shape)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        return np.clip(actions + noise, -self.act_range, self.act_range)

###############################################

class TD3Critic(object):
    def __init__(self, state_dim, action_dim, lr, tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau

        self.model1 = self.buildNetwork()
        self.model2 = self.buildNetwork()
        self.target_model1 = self.buildNetwork()
        self.target_model2 = self.buildNetwork()
        
        self.target_model1.set_weights(self.model1.get_weights())
        self.target_model2.set_weights(self.model2.get_weights())

        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=100000,
            decay_rate=0.99
        )
        self.optimizer1 = Adam(learning_rate=self.lr_schedule)
        self.optimizer2 = Adam(learning_rate=self.lr_schedule)
        

    def buildNetwork(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        x = Concatenate()([state_input, action_input])
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1)(x)
        return Model(inputs=[state_input, action_input], outputs=output)
    

    @tf.function
    def predict(self, states, actions):
        return self.model1([states, actions]), self.model2([states, actions])
    

    @tf.function
    def target_predict(self, states, actions):
        return self.target_model1([states, actions]), self.target_model2([states, actions])
    

    @tf.function
    def transferWeights(self):
        for a, b in zip(self.target_model1.variables, self.model1.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)
        for a, b in zip(self.target_model2.variables, self.model2.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)

###############################################

class TD3Agent(object):
    def __init__(
        self, state_dim, action_dim, action_min, action_max, 
        memory_size, batch_size, gamma, a_lr, c_lr, tau, max_steps, 
        env_name, policy_noise=0.2, noise_clip=0.5, policy_freq=2
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_steps = max_steps
        self.env_name = env_name
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.memory = PrioritizedReplayBuffer(memory_size, batch_size)

        self.actor = TD3Actor(
            inp_dim=self.state_dim, 
            out_dim=self.action_dim,
            act_range=self.action_max, 
            lr=a_lr, 
            tau=self.tau,
        )

        self.critic = TD3Critic(
            state_dim=self.state_dim, 
            action_dim=self.action_dim,
            lr=c_lr, 
            tau=self.tau,
        )

        self.total_it = 0
        self.create_plot()


    def create_plot(self):
        # Create a figure for SOM activation visualization
        self.fig = plt.figure()

        self.returns = self.fig.add_subplot(211)
        self.returns.title.set_text('Retruns')

        self.n_steps = self.fig.add_subplot(212)
        self.n_steps.title.set_text('N Steps')

        self.fig.show()
        return


    def update_plots(self, returns, n_steps):
        # Update the SOM activation plot
        self.returns.plot(np.arange(len(returns)), returns)

        self.n_steps.plot(np.arange(len(returns)), n_steps)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return


    @tf.function
    def policy(self, state):
        action = self.actor.predict(tf.expand_dims(state, 0))
        action = self.actor.add_noise(tf.expand_dims(action, 0))
        return tf.clip_by_value(action[0], self.action_min, self.action_max)


    def learn(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)
        self.replay_memory()


    @tf.function
    def update_nets(self, weights, states, actions, rewards, next_states, dones):
        weights = tf.cast(weights, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            # Select action according to policy and add clipped noise
            noise = tf.random.normal(tf.shape(actions), stddev=self.policy_noise)
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor.target_predict(next_states) + noise
            next_actions = tf.clip_by_value(next_actions, self.action_min, self.action_max)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic.target_predict(next_states, next_actions)
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic.predict(states, actions)

            # Compute critic loss
            critic_loss = tf.reduce_mean(tf.square(target_Q - current_Q1)) + tf.reduce_mean(tf.square(target_Q - current_Q2))

        # Optimize the critic
        critic_grad1 = tape.gradient(critic_loss, self.critic.model1.trainable_variables)
        critic_grad2 = tape.gradient(critic_loss, self.critic.model2.trainable_variables)
        self.critic.optimizer1.apply_gradients(zip(critic_grad1, self.critic.model1.trainable_variables))
        self.critic.optimizer2.apply_gradients(zip(critic_grad2, self.critic.model2.trainable_variables))

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(self.critic.model1([states, self.actor.predict(states)]))

            actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))

            # Update the frozen target models
            self.actor.transferWeights()
            self.critic.transferWeights()

        return target_Q, tf.minimum(current_Q1, current_Q2)



    def replay_memory(self):
        if not self.memory.isMin(): return  # Not enough samples in the buffer
        
        # Sample from replay buffer and perform learning update
        experiences, indices, weights = self.memory.sample()
        
        states = tf.convert_to_tensor([exp[0] for exp in experiences], dtype=tf.float32)
        actions = tf.convert_to_tensor([exp[1] for exp in experiences], dtype=tf.float32)
        rewards = tf.convert_to_tensor([exp[2] for exp in experiences], dtype=tf.float32)
        next_states = tf.convert_to_tensor([exp[3] for exp in experiences], dtype=tf.float32)
        dones = tf.convert_to_tensor([exp[4] for exp in experiences], dtype=tf.float32)

        #self.actor.vae.train_step(tf.concat([states, next_states], axis=0))
        y, critic_value = self.update_nets(weights, states, actions, rewards, next_states, dones)
        self.total_it += 1

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
                    reward = -100
                    done = True

                self.learn(observation, action.numpy(), reward, new_observation, done)
                observation = new_observation
                score += reward
                steps += 1

            scores_history.append(score)
            steps_history.append(steps)
            self.update_plots(scores_history, steps_history)
            
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
    
###############################################

name = "BipedalWalker-v3"
env = gym.make(name, hardcore=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_min = env.action_space.low
action_max = env.action_space.high

memory_size = 1000000
batch_size = 256
gamma = 0.99
a_lr = 1e-4
c_lr = 1e-3
tau = 1e-3
max_steps = 2000

agent = TD3Agent(
    state_dim, action_dim, action_min, action_max, 
    memory_size, batch_size, gamma, a_lr, c_lr, tau, 
    max_steps, name
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