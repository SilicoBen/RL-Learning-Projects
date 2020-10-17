import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import DeepQNetwork
from buffer import MemoryBuffer

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size = 100000, eps_end = 0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon # used for exploration 
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.memory = MemoryBuffer(max_mem_size, input_dims)

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

    def choose_action(self, observation):
        if  np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        # When we start we have a memory buffer full of 0s you can deal with this in different ways
        # 1. let the agent play a bunch of games just randomly selecting actions until buffer full
        # 2. start learning once you've filled up the batch_size not the whole memory (below)
        if self.memory.mem_cntr < self.batch_size:
            return 
        
        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        # Convert numpy arrays from memory to tensors
        state_batch = T.tensor(states, dtype=T.float).to(self.Q_eval.device)
        reward_batch = T.tensor(rewards, dtype=T.float).to(self.Q_eval.device)
        new_state_batch = T.tensor(new_states, dtype=T.float).to(self.Q_eval.device)
        terminal_batch = T.tensor(dones).to(self.Q_eval.device)
        action_batch = actions

        # We also need a batch index in order to perform the proper array slicing without this 
        # we don't get the right thing
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        # now we perform the feedforward 
        # our goal is move the agents estimates of the value of the current state
        # towards the maximimum value for the next state (i.e tilt the agnet toward selecting maximal actions)
        
        self.Q_eval.optimizer.zero_grad() # Note: when using pytorch you need to zero out your optimizer (not necessary in other packages)
        
        # we want to get the value of the actions that we actually took 
        # batch index is used here to do the de referencing (slice up)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] 
        # this is where the target network would go if you were doing full deepQ
        q_next = self.Q_eval.forward(new_state_batch) # no dereferencing here. 
        q_next[terminal_batch] = 0.0

        # calculate target values (we update our estimates towards these)
        # we use the purly greedy action to update our loss function 
        # this is the off policy part of the algorithm since we update loss function 
        # with greedy but we use episolon greed policy to select our actual action
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]  # note: T.max returns tuple (max_value, index) (i.e. greedy action)
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # Each time we learn we need to decrese the epsilon value by 1 unit of decrement 
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


        