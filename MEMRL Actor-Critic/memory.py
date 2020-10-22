from __future__ import divition, print_function
import numpy as np
from futils import softmax

class EpisodicMemory(object):
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
        self.n_actions = n_actions

    # state_ = new state
    def encode_memory(self, state, action, reward, state_, done):
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
    def recall_memory(self, state_representation):
        
        # Sample memories
        if len(self.state_memory) == 0:  # check if there are memories stored
            random_policy = softmax(np.zeros(self.n_actions))
            return random_policy
        else:
            lin_act, similarity = self.cosine_sim(state_representation) # returns the most similar key, as well as the cosine similarity measure
            memory       = np.nan_to_num(self.cache_list[lin_act][0])
            deltas       = memory[:,0]
            if self.use_pvals:
                times = abs(timestep - memory[:, 1])
                pvals = self.make_pvals(times, envelope=envelope)
                policy = softmax( similarity*np.multiply(deltas,pvals), T=mem_temp)
            else:
                policy = softmax( similarity*deltas, T=mem_temp)
            return policy

    def cosine_sim(self, representation): #
        # make list of memory keys
        mem_cache = np.asarray(list(self.state_memory.keys()))

        entry = np.asarray(representation)
        # compute cosine similarity measure
        mqt = np.dot(mem_cache, entry)
        norm = np.linalg.norm(mem_cache, axis=1) * np.linalg.norm(entry)
        cosine_similarity = mqt / norm

        lin_act = mem_cache[np.argmax(cosine_similarity)]
        return  tuple(lin_act), max(cosine_similarity)
        