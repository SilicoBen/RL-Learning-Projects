# Goal - get robust learning in a continuous environment
# can also use algorithms like Deep Deterministic Policy Gradient (DDPG) or 
# TD3 for continuous action environments
# TD3 and SAC are on par, while TDPG falls a bit short
# TD3 and DDPG output actions directly
# SAC outputs mean and SD for a normal distribution which is then sampled to get action
# SAC uses a maximum entropy framework - 
## for this we want scale the cost function to encourage exploration 
## but does so that is robust to random seeds in the environment
## and accounts for episode-to-episode variation 
### This smooths out the Rewards over Time graph. 
### this is because it not only maximizes the reward overtime but also the 
### stocasticiy (randomness/entropy) of how the agent behaves

###NETWORKS###
# 1. CriticNetwork
### input = state, action
### output = how good was the action
# 2. ValueNetwork
### describes how valuable a state is
# 3. ActorNetwork 
### output = mean and standard deviation, which we sample to get action

# using these networks we can figure out the best action and what is the best
# sequence of states to access so we can know what actions to take over time

import os  # handles model checkpointing 
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, 
            name = 'critic', chkpt_dir = 'tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # now we define the neural network 
        # the critic evaluates the value of a state-action pair
        # so we want to incorporate the action right from the very beginning of the input
        # note: in deep deterministic policy gradients you can just pass in the state to 
        # the first input layer and then pass in the action later
        # hidden layers
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # output layer
        self.q = nn.Linear(self.fc2_dims, 1) # outputs scaler quantity

        # now we need an optimizer to optimize the parameters of our NN
        # self.parameters comes from nn.Module
        # set learning rate to beta
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cpu')

        self.to(self.device)

    # next we describe how to feed a state-action pair through the network
    def forward(self, state, action):
        # action_value is the feedforward of the concatination of state-action along the batch dimension
        # through our first fully connected layer
        action_value = self.fc1(T.cat([state, action], dim=1))
        # relu is the activation function of the network  (returns max(0, x))
        ## e.g. if input = -2 output = 0, if input = 2 output = 2. 
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    # next we need functions for saving/loading model checkpoints 
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, 
            name='value', chkpt_dir='tmp/sac'):
            super(ValueNetwork, self).__init__()
            self.input_dims = input_dims
            self.fc1_dims = fc1_dims
            self.fc2_dims = fc2_dims
            self.name = name
            self.checkpoint_dir = chkpt_dir
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

            # network
            # hidden layers
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
            # output layer
            self.v = nn.Linear(self.fc2_dims, 1)

            self.optimizer = optim.Adam(self.parameters(), lr=beta)
            self.device = T.device('cpu')

            self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))

# The ActorNetwork is more complex because instead of just handling feedforward
# we need to handle sampling and probability distributions 
# We need max_actions is that our policy sampling is going to be restrained to (-1,1)
# however an environment might have an action bound that is greater than (-1,1)
# so we need to multiply the output of the network by the max_action in order to get 
# the action value
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        self.reparam_noise = 1e-6  # repram_noise makes sure we dont take log(0)
        
        # define the network
        # hidden layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # output layers
        # mu/sigma have as many outputs as we have components to our actions
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # here we have to do something a little different that previous networks
        # we need to clamp our sigma so that our distribution isn't arbitrarily broad
        # we want sigma to be some finite and constrained value
        # in the paper they use range (-20, 2) however, pytorch doesn't like 0
        # instead we're going to use reparam to make sure that we don't get 0
        # we can also have a sigmoidal function to get somewhere between 0-1
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    # Recall: a policy is a probability distribution that tells you what the prob
    # of selecting any action in your action space given a state or set of states
    # in a discrete space we just assign  
    def sample_normal(self, state, reparameterize = True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # rsample allows the sample to be differentiable which allows us
        # to reparameterize it
        if reparameterize:
            actions = probabilities.rsample() 
        else:
            actions = probabilities.sample()

        # get the action for the agent using a tan-hyperbolic function
        # we multiply it by max action because our action space is between 
        # -1-1 but our action space might be have actions above or below that range
        # .to(self.device) ensure the output is in the same format as our other 
        # outputs (i.e. cuda tensors)
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        # we need the log prob for the calculation of our loss function
        # loss function is used to update the weights of our NN
        # we take this on the actions not the action that we calc above
        log_probs = probabilities.log_prob(actions)
        # for more info about this line see the paper
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise) # add rn so we don't ever get 0
        # take the sum we need a scaler quantity
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))


        
        

        

        








