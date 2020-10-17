import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # the * is a way of unpacking an array of any dimension
        # this makes our code more functional since we can now pass 
        # any number of different types of arrays to the network
        # and it will still work
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # a deep Q network provides an estimate of the value of each action
        # given some set of states
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Deep Q is essentially linear regression, where we fit a line 
        # to the delta between target value and output of the network
        self.loss = nn.MSELoss()

        # Use GPU if there is one available and send network to device 
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        # Note: we don't want to activate our third layer here (e.g. using relu)
        # this is because we want to get the agents raw estimate 
        # if the present value of future rewards is negative relu wont help us
        # It could also be the case that our estimate is greater than 1
        return actions



