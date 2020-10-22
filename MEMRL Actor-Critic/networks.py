import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

### here we create two different classes one for the actor and one for the critic network
# the actor network tells the agent what actions to take given the policy approximation
# the other network is the critic which says whether the action is good or bad based on
# its approximation of the value of the state-action pair (learns values of actions)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name = "critic", chkpt_dir = "tmp/sac"):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")

        # network connections 
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, self.n_actions)

        # optimizer - parameters come fomr nn.Module 
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device) # transforms np float array to cuda tensor
        # feed observation through layers
        action_value = F.relu(self.fc1(state))
        action_value = F.relu(self.fc2(action_value))
        q = self.q(action_value) # this is activated later when we select an action 

        return q

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

        # network connections 
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, self.n_actions)

        # optimizer - parameters come fomr nn.Module 
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
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
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.linear(self.fc2_dims, self.n_actions)
        # output layers
        # mu/sigma have as many outputs as we have components to our actions
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        state = T.Tensor(state).to(self.device) # transforms np float array to cuda tensor
        # feed observation through layers
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = self.fc3(action) # this is activated later when we select an action 

        return action
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))