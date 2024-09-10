import gym
import tensorflow as tf, matplotlib.pyplot as plt, numpy as np

from tensorflow.keras.layers import Dense, BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple

global_seed = 42
tf.random.set_seed(global_seed)
np.random.seed(global_seed)

##########################################

class OUActionNoise:
    def __init__(self, mean, sigma=0.5, theta=0.2, dt=0.1, x0=None):
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.theta = tf.constant(theta, dtype=tf.float32)
        self.dt = tf.constant(dt, dtype=tf.float32)
        self.x0 = x0
        self.reset()


    @tf.function
    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.sigma * tf.sqrt(self.dt) * tf.random.normal(self.mean.shape)
        self.x_prev = x
        return x


    def reset(self):
        if self.x0 is not None:
            self.x_prev = tf.constant(self.x0, dtype=tf.float32)
        else:
            self.x_prev = tf.zeros_like(self.mean)

##########################################

class ReplayBuffer(object):
    def __init__(self, size, minibatch_size = None):
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState()
        self.max_size = size
        
    #--------------------------------------------------------------------------------    
    def append(self, state, action, reward, next_state, context, next_context, done):
        if self.size() == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, next_state, context, next_context, int(done)])
    
    #--------------------------------------------------------------------------------    
    def sample(self):
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]
    
    #--------------------------------------------------------------------------------    
    def size(self):
        return len(self.buffer)
    
    #--------------------------------------------------------------------------------
    def isMin(self):
        return (self.size() >= self.minibatch_size)
    
    #--------------------------------------------------------------------------------
    def empties(self):
        self.buffer.clear()
    
    #--------------------------------------------------------------------------------
    def getEpisode(self):
        return self.buffer
    
##########################################

class SOM:
    def __init__(self, m, n, dim, n_iterations, alpha, sigma=None):
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iterations = n_iterations
        
        if sigma is None:
            sigma = max(m, n) / 2.0
        
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.sigma = tf.Variable(sigma, dtype=tf.float32)
        
        self.weights = tf.Variable(tf.random.uniform([m * n, dim]))
        self.locations = tf.constant([(i, j) for i in range(m) for j in range(n)], dtype=tf.float32)
    
    
    @tf.function
    def get_bmu(self, input_vector):
        distances = tf.reduce_sum(tf.square(self.weights - input_vector), axis=1)
        bmu_index = tf.argmin(distances)
        return bmu_index
    

    @tf.function
    def update_weights(self, input_vector, bmu_index):
        bmu_location = tf.gather(self.locations, bmu_index)
        distance_sq = tf.reduce_sum(tf.square(self.locations - bmu_location), axis=1)
        neighborhood = tf.exp(-distance_sq / (2 * tf.square(self.sigma)))
        
        learning = tf.expand_dims(neighborhood * self.alpha, axis=1) * (input_vector - self.weights)
        self.weights.assign_add(learning)
    

    @tf.function
    def train_step(self, input_vector):
        bmu_index = self.get_bmu(input_vector)
        self.update_weights(input_vector, bmu_index)
    

    @tf.function
    def train(self, input_data):
        for i in range(self.n_iterations):
            for vector in input_data:
                self.train_step(vector)


    @tf.function
    def get_output(self, input_vector):
        distances = tf.reduce_sum(tf.square(self.weights - input_vector), axis=1)
        return tf.reshape(distances, [self.m, self.n])
    
##########################################

class InstinctiveWeights:
    def __init__(self, som_shape: Tuple[int, int]):
        self.som_shape = som_shape
        self.num_neurons = np.prod(self.som_shape)
        self.weights = tf.Variable(tf.zeros(int((self.num_neurons*(self.num_neurons+1))/2), dtype=tf.float32))
        self.activations = tf.Variable(tf.zeros(self.som_shape, dtype=tf.float32))


    def __neuron_to_position(self, neuron: Tuple[int, int]) -> int:
        i, j = neuron
        return int((i*self.som_shape[1]) + j)


    def __positions_to_index(self, position: Tuple[int, int]) -> int:
        i, j = position
        return int(((i*(i+1))/2) + j)


    def __get_tril_positions(self, n1: Tuple[int, int], n2: Tuple[int, int]) -> Tuple[int, int]:
        n1_pos_aux = self.__neuron_to_position(n1)
        n2_pos_aux = self.__neuron_to_position(n2)

        n1_pos = tf.maximum(n1_pos_aux, n2_pos_aux)
        n2_pos = tf.minimum(n1_pos_aux, n2_pos_aux)

        return n1_pos, n2_pos


    def reinforce_connection(self, n1: Tuple[int, int], n2: Tuple[int, int]) -> None:
        weight_position = self.__get_tril_positions(n1, n2)
        array_index = self.__positions_to_index(weight_position)
        value = self.weights[array_index]

        value += 0.1
        value = tf.clip_by_value(value, 0, 1)

        self.weights[array_index] = value

    
    @tf.function
    def step(self):
        self.weights.assign(self.weights * 0.995)

    
    @tf.function
    def get_weight_matrix(self):
        full_matrix = tf.zeros((self.num_neurons, self.num_neurons), dtype=tf.float32)
        
        indices = tf.constant([(i, j) for i in range(self.num_neurons) for j in range(i + 1)])
        updates = self.weights[:tf.shape(indices)[0]]
        
        full_matrix = tf.tensor_scatter_nd_update(full_matrix, indices, updates)
        full_matrix = full_matrix + tf.transpose(full_matrix) - tf.linalg.diag(tf.linalg.diag_part(full_matrix))
        
        return full_matrix
    
##########################################

class InstinctiveLayer:
    def __init__(
        self, som_shape: Tuple[int, int], func_x_max: float, func_x_drop: float, 
        func_y_max: float, func_y_min: float, act_threshold: float,
    ):
        self.som_shape = som_shape
        self.act_threshold = act_threshold
        self.func_x_max = func_x_max
        self.func_x_drop = func_x_drop
        self.func_y_max = func_y_max
        self.func_y_min = func_y_min

        self.charges = tf.Variable(tf.ones(self.som_shape) * (self.func_x_max/100), dtype=tf.float32)


    @tf.function
    def get_act_array(self):
        return tf.reshape(self.get_activations(), [-1])


    @tf.function
    def get_active(self):
        act = self.get_activations()
        active = tf.where(tf.equal(act, 1))
        return tf.cast(active, tf.int32)


    @tf.function
    def step(self, weights, som_act):
        charge_delta = tf.reshape(tf.matmul(tf.reshape(self.charges * self.get_activations(), [1, -1]), weights), self.charges.shape)
        charge_delta = tf.clip_by_value(charge_delta, 0, self.func_x_max/50)
        
        som_act *= self.func_x_max/50

        new_charges = self.charges + charge_delta + som_act

        # Apply conditions using tf.where
        condition1 = tf.greater(new_charges, self.func_x_drop)
        condition2 = tf.less_equal(new_charges, self.func_x_drop)
        condition3 = tf.greater(new_charges, self.func_x_max)
        condition4 = tf.less(new_charges, 0)

        new_charges = tf.where(condition1, new_charges + self.func_x_max/100, new_charges)
        new_charges = tf.where(condition2, new_charges - self.func_x_max/100, new_charges)
        new_charges = tf.where(condition3, tf.zeros_like(new_charges), new_charges)
        new_charges = tf.where(condition4, tf.zeros_like(new_charges), new_charges)

        self.charges.assign(new_charges)


    @tf.function
    def __apply_activation_function(self, charges):
        condition1 = tf.less_equal(charges, self.func_x_drop)
        condition2 = tf.logical_and(tf.greater(charges, self.func_x_drop), tf.less_equal(charges, self.func_x_max))

        result1 = (self.func_y_max / self.func_x_drop**2) * charges**2
        result2 = self.func_y_min + (0 - self.func_y_min) / (self.func_x_max - self.func_x_drop) * (charges - self.func_x_drop)

        return tf.where(condition1, result1, tf.where(condition2, result2, tf.zeros_like(charges)))


    @tf.function
    def reset_charges(self):
        self.charges.assign(tf.ones(self.som_shape) * (self.func_x_max/100))


    @tf.function
    def get_activations(self):
        activations = self.__apply_activation_function(self.charges)
        activations = tf.clip_by_value(activations, 0, self.act_threshold)
        activations = activations / self.act_threshold
        activations = tf.where(tf.greater(self.charges, self.func_x_drop), tf.ones_like(activations) * (self.func_x_max/100), activations)
        return activations
    
    
    def plot_act_func(self):
        # Generate x values
        x = np.linspace(0, self.func_x_max, 1000)
        y = self.__apply_activation_function(x)

        # Plot the function
        plt.plot(x, y, label="Piecewise Function with Repetition")
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Piecewise Function: Exponential Growth, Peak, Drop, Linear Growth, and Repeat')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=self.func_y_max, color='gray', linestyle='--')
        plt.axhline(y=self.func_y_min, color='gray', linestyle='--')
        plt.axvline(x=self.func_x_drop, color='gray', linestyle='--')
        plt.axvline(x=self.func_x_max, color='gray', linestyle='--')
        plt.axhline(y = 0, color = 'r', linestyle = '-') 
        plt.axhline(y = self.act_threshold, color = 'r', linestyle = '-') 
        plt.show()

##########################################

class InstinctiveNetwork:

    def __init__(self, som_dims: Tuple[int, int], input_dim: int, som_kwargs: dict, inst_net_kwargs: dict):
        self.som_dims = som_dims
        self.input_dim = input_dim
        self.som = SOM(*self.som_dims, self.input_dim, **som_kwargs)
        self.inner_weights = InstinctiveWeights(self.som_dims)
        self.inner_layer = InstinctiveLayer(som_shape=self.som_dims, **inst_net_kwargs)
        self.last_winner = None


    @tf.function
    def train_som(self, data):
        self.som.train(data)


    @tf.function
    def reinforce_connection(self, data):
        som_winner = self.som.get_bmu(data)
        if self.last_winner is not None:
            self.inner_weights.reinforce_connection(self.last_winner, som_winner)
        self.last_winner = som_winner
    

    def reset_charges(self):
        self.inner_layer.reset_charges()
        self.last_winner = None


    @tf.function
    def get_output(self, data, reinforce=True):
        if reinforce:
            self.inner_weights.step()
            self.reinforce_connection(data)

        som_act = self.som.get_output(data)
        som_act = 1 - ((som_act - tf.reduce_min(som_act)) / (tf.reduce_max(som_act) - tf.reduce_min(som_act)))
        
        som_act_dist = tf.where(som_act < 0.9, tf.zeros_like(som_act), som_act)

        weights = self.inner_weights.get_weight_matrix()
        weights = tf.cast(weights, tf.float32)
        self.inner_layer.step(weights, som_act_dist)

        active = self.inner_layer.get_act_array()
        return active, som_act
    
##########################################

class Actor(Model):
    def __init__(self, s_inp_dim, s_fc1_dim, con_inp_dim, con_fc1_dim, fc2_dim, fc3_dim, fc4_dim, fc5_dim, out_dim, act_range, lr, tau):
        super(Actor, self).__init__()
        self.act_range = act_range
        self.tau = tau
        
        self.s_fc1 = Dense(s_fc1_dim, activation='relu')
        self.s_bn1 = BatchNormalization()
        
        self.con_fc1 = Dense(con_fc1_dim, activation='relu')
        self.con_bn1 = BatchNormalization()
        
        self.fc2 = Dense(fc2_dim, activation='relu')
        self.bn2 = BatchNormalization()
        
        self.fc3 = Dense(fc3_dim, activation='relu')
        self.bn3 = BatchNormalization()

        self.fc4 = Dense(fc4_dim, activation='relu')
        self.bn4 = BatchNormalization()

        self.fc5 = Dense(fc5_dim, activation='relu')
        self.bn5 = BatchNormalization()
        
        self.out = Dense(out_dim, activation='tanh')
        
        self.optimizer = Adam(learning_rate=lr)


    @tf.function
    def call(self, state, context):
        s = self.s_fc1(state)
        s = self.s_bn1(s)
        
        c = self.con_fc1(context)
        c = self.con_bn1(c)
        
        x = tf.concat([s, c], axis=1)
        
        x = self.fc2(x)
        x = self.bn2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)

        x = self.fc4(x)
        x = self.bn4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        
        x = self.out(x)
        return x * self.act_range


    @tf.function
    def transfer_weights(self, target_model):
        for a, b in zip(target_model.variables, self.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)

##########################################

class Critic(Model):
    def __init__(self, state_inp_dim, state_fc1_dim, action_inp_dim, action_fc1_dim, conc_fc1_dim, conc_fc2_dim, conc_fc3_dim, conc_fc4_dim, out_dim, lr, tau):
        super(Critic, self).__init__()
        self.tau = tau
        
        self.s_fc1 = Dense(state_fc1_dim, activation='relu')
        self.s_bn1 = BatchNormalization()
        
        self.a_fc1 = Dense(action_fc1_dim, activation='relu')
        self.a_bn1 = BatchNormalization()
        
        self.fc1 = Dense(conc_fc1_dim, activation='relu')
        self.bn1 = BatchNormalization()
        
        self.fc2 = Dense(conc_fc2_dim, activation='relu')
        self.bn2 = BatchNormalization()

        self.fc3 = Dense(conc_fc3_dim, activation='relu')
        self.bn3 = BatchNormalization()

        self.fc4 = Dense(conc_fc4_dim, activation='relu')
        self.bn4 = BatchNormalization()
        
        self.out = Dense(out_dim, activation='linear')
        
        self.optimizer = Adam(learning_rate=lr)

    @tf.function
    def call(self, state, action):
        s = self.s_fc1(state)
        s = self.s_bn1(s)
        
        a = self.a_fc1(action)
        a = self.a_bn1(a)
        
        x = tf.concat([s, a], axis=1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.bn3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        
        x = self.out(x)
        return x


    @tf.function
    def transfer_weights(self, target_model):
        for a, b in zip(target_model.variables, self.variables):
            a.assign(self.tau * b + (1 - self.tau) * a)

##########################################

class DDPGAgent(object):
    def __init__(
        self, state_dim, action_dim, action_min, action_max, 
        memory_size, batch_size, gamma, a_lr, c_lr, tau, epsilon, 
        epsilon_decay, epsilon_min, max_steps, env_name
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
        self.max_steps = max_steps
        self.env_name = env_name

        self.noise = OUActionNoise(mean=np.zeros(action_dim), sigma=0.5, theta=0.2)

        #Creates the Replay Buffer
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        # creates instinctive network
        som_dim = (15, 15)
        self.inst_net = InstinctiveNetwork(
            som_dims = som_dim, 
            input_dim = self.state_dim,
            som_kwargs = {
                'n_iterations': 1,
                'alpha': 5e-3,
                'sigma': 1,
            }, 
            inst_net_kwargs = {
                'func_x_max': 50,
                'func_x_drop': 35,
                'func_y_max': 10,
                'func_y_min': -7,
                'act_threshold': 4,
            }
        )

        self.actor, self.actor_target = [Actor(
            s_inp_dim=self.state_dim, 
            s_fc1_dim=256,
            con_inp_dim=np.prod(som_dim), 
            con_fc1_dim=256,
            fc2_dim=512, 
            fc3_dim=256,
            fc4_dim=128,
            fc5_dim=64,
            out_dim=self.action_dim, 
            act_range=self.action_max, 
            lr=self.a_lr, 
            tau=self.tau,
        ) for _ in range(2)]
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic, self.critic_target = [Critic(
            state_inp_dim=self.state_dim, 
            state_fc1_dim=256, 
            action_inp_dim=self.action_dim, 
            action_fc1_dim=128,
            conc_fc1_dim=512, 
            conc_fc2_dim=256,
            conc_fc3_dim=128,
            conc_fc4_dim=64,
            out_dim=1,
            lr=self.c_lr, 
            tau=self.tau,
        ) for _ in range(2)]
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.create_plot()
        return


    def create_plot(self):
        # Create a figure for SOM activation visualization
        self.fig = plt.figure()

        self.returns = self.fig.add_subplot(221)
        self.returns.title.set_text('Retruns')

        self.n_steps = self.fig.add_subplot(223)
        self.n_steps.title.set_text('N Steps')

        self.som_val = self.fig.add_subplot(222)
        self.som_val.title.set_text('SOM val')

        self.som_act = self.fig.add_subplot(224)
        self.som_act.title.set_text('SOM Activation')

        self.fig.show()
        return


    def update_plots(self, returns=None, n_steps=None, som_val=None, som_act=None):
        # Update the SOM activation plot
        if returns is not None:
            self.returns.plot(np.arange(len(returns)), returns)

        if n_steps is not None:
            self.n_steps.plot(np.arange(len(returns)), n_steps)

        if som_val is not None:
            self.som_val.imshow(som_val)

        if som_act is not None:
            self.som_act.imshow(np.reshape(som_act, np.shape(som_val)))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return


    @tf.function
    def policy(self, state, context, explore=True):
        action = self.actor(tf.expand_dims(state, 0), tf.expand_dims(context, 0))[0]
        if explore:
            if tf.random.uniform(()) < self.epsilon:
                noise = self.noise()
                action += noise
        return tf.clip_by_value(action, self.action_min, self.action_max)


    @tf.function
    def learn(self, states, actions, rewards, next_states, contexts, next_contexts, done):
        # Train SOM using TensorFlow operations
        all_states = tf.concat([states, next_states], axis=0)
        self.inst_net.train_som(all_states)

        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_states, next_contexts)
            target_q_values = self.critic_target(next_states, target_actions)
            y = rewards + self.gamma * target_q_values * (1 - done)
            
            q_values = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(y - q_values))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(states, contexts)
            critic_value = self.critic(states, actions)
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.actor.transfer_weights(self.actor_target)
        self.critic.transfer_weights(self.critic_target)

        return actor_loss, critic_loss


    def act(self):
        #Reset the envirorment
        env2 = gym.make(self.env_name, hardcore=True, render_mode='human')
        state, _ = env2.reset()
        self.inst_net.reset_charges()
        done = False
        step = 0
        
        while not done:
            env2.render()
            context, _ = self.inst_net.get_output(state)
            action = self.policy(state, context, explore=False)
            #self.update_plots(som_act, self.inst_net.inner_layer.get_activations())
            state, reward, done, _, _ = env2.step(action.numpy())
            step += 1
            done = done or (step > self.max_steps)
        
        env2.close()
        return


    def train(self, env, num_episodes, verbose, verbose_num, end_on_complete, complete_num, complete_value, act_after_batch, plot_act):
        scores_history = []
        steps_history = []

        print("BEGIN\n")
        complete = 0

        for episode in range(num_episodes):
            state, _ = env.reset()
            self.inst_net.reset_charges()
            done = False
            score = 0
            steps = 0

            while not done:
                context, som_val = self.inst_net.get_output(state)
                action = self.policy(state, context)
                if plot_act: self.update_plots(som_val=som_val.numpy(), som_act=context.numpy())

                if verbose:
                    print("\r                                                          ", end="")
                    print(f"\rEpisode: {str(episode+1)} \tStep: {str(steps)} \tReward: {str(score)}", end="")
                
                next_state, reward, done, _, _ = env.step(action.numpy())
                next_context, _ = self.inst_net.get_output(next_state)
                
                self.memory.append(state, action.numpy(), reward, next_state, context, next_context, done)
                
                if self.memory.isMin():
                    experiences = self.memory.sample()
                    states, actions, rewards, next_states, contexts, next_contexts, dones = [np.array([exp[i] for exp in experiences]) for i in range(7)]
                    
                    self.learn(
                        tf.convert_to_tensor(states, dtype=tf.float32),
                        tf.convert_to_tensor(actions, dtype=tf.float32),
                        tf.convert_to_tensor(rewards, dtype=tf.float32),
                        tf.convert_to_tensor(next_states, dtype=tf.float32),
                        tf.convert_to_tensor(contexts, dtype=tf.float32),
                        tf.convert_to_tensor(next_contexts, dtype=tf.float32),
                        tf.convert_to_tensor(dones, dtype=tf.float32)
                    )
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                state = next_state
                score += reward
                steps += 1
                done = done or (steps > self.max_steps)


            scores_history.append(score)
            steps_history.append(steps)
            self.update_plots(returns=scores_history, n_steps=steps_history)
            
            if(score >= complete_value):
                complete += 1
                if end_on_complete and complete >= complete_num: break
            
            if((episode+1)%verbose_num == 0):
                print("\r                                                                                                          ", end="")
                print(f'''\rEpisodes: {episode+1}/{num_episodes}\n\tTotal reward: {np.mean(scores_history[-verbose_num:])} +- {np.std(scores_history[-verbose_num:])}\n\tNum. steps: {np.mean(steps_history[-verbose_num:])} +- {np.std(steps_history[-verbose_num:])}\n\tCompleted: {complete}\n--------------------------''')
                if act_after_batch: self.act()
                complete = 0

        print("\nFINISHED")
        
        return scores_history, steps_history


    def save(self, path):
        self.actor.saveModel(path)
        self.critic.saveModel(path)
        return


    def load(self, a_path, c_path):
        self.actor.loadModel(a_path)
        self.critic.loadModel(c_path)
        return

##########################################

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
c_lr = 1e-3
tau = 1e-3
epsilon = 1
epsilon_decay = 0.9999
epsilon_min = 0.4
max_steps = 2000

agent = DDPGAgent(
    state_dim, action_dim, action_min, action_max, 
    memory_size, batch_size, gamma, a_lr, c_lr, tau, 
    epsilon, epsilon_decay, epsilon_min, max_steps, name
)

num_episodes = 3000
verbose = True
verbose_num = 5
end_on_complete = True
complete_num = 10
complete_value = 300
act_after_batch = True

agent.train(
    env, num_episodes, verbose, 
    verbose_num, end_on_complete, 
    complete_num, complete_value, 
    act_after_batch, plot_act=False
)
