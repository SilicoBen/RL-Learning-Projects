# https://www.youtube.com/watch?v=ioidsRlf79o

import numpy as np

# Start by creating the agents memory (replay buffer)
# we'll use numpy arrays for this (easier to understand)
# We're designing this for a continuous environment
# max_size because we don't want buffer to be unbounded! (typically = 1M trans)
# input shape = observation dimensionality from the environment
# n_actions = # of complements (because continuous)
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0  # holds position of first available memory
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))   # keeps track of the states that result after actions
        self.action_memory = np.zeros((self.mem_size, n_actions))  # num of complements of the actions
        self.reward_memory = np.zeros(self.mem_size)  # array of scalers - keeps track of rewards agent recieves with each step of the env
        # we need terminal memory is because the value of the terminal state is identically 0
        # so we need to store the done flags from the env as a way of setting the values of the terminal state to 0
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # state_ = new state
    def store_transition(self, state, action, reward, state_, done):
        # finds index of first available memory
        index = self.mem_cntr % self.mem_size 

        #  set the numpy arrays at the index = parameters we passed in
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # function that handles our buffer 
    def sample_buffer(self, batch_size):
        # find out how many memories we've stored in our buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        # select a sequence of integers between and the max memory
        batch = np.random.choice(max_mem, batch_size)

        # Sample memories 
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


