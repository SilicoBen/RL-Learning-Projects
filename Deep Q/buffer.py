import numpy as np

class MemoryBuffer(object):
    
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.input_dims = input_dims
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) # pytorch is particular about data types and enforces type checking 
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)  # we use an int because we have a discrete action space. 
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) 
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) # if terminal state the game is done there are no actions! future value of terminal state = 0

    def store_transition(self, state, action, reward, state_, done):
        # finds position of the first unoccupied memory slot
        # we use the moduluous becuase it will wrap back around once it 
        # gets to the end of the list it will go to 0 this way we can just let the cntr run
        # this allows us to rewrite the agents earliest memories when 
        # we run out of space in the buffer
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # find out how many memories are in our buffer 
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # select a sequence of integers of batch_size between 0 and max size
        # we don't want to select the same memories more than once (a problem when buffer is low)
        batch = np.random.choice(max_mem, batch_size, replace =False)

        # sample memories
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones