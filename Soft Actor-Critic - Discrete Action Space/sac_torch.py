import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    # Notes: reward scale is how we account for the entropy in the framework
    # this depends on the number of action dimensions of the environment
    # may need to increase with higher dimensionality 
    # tau is the factor by which we are going modulate the parameters of our 
    # target-value network - instead of doing a hard copy of our value network
    # we'll use tau to do a soft copy which retunes the parameters
    # this is similar to TD3 and DDPG
    def __init__(self, alpha=0.003, beta=0.003, input_dims=[8], tau=0.005,
            env=None, gamma=0.99, n_actions=4, max_size=100000, batch_size=64, reward_scale=2):
        self.gamma = gamma 
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        #get our networks
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims)
        self.target_value = ValueNetwork(beta, input_dims,name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # Remember we need to activate fc3 + probabilities need to sum to 1
        # to do this we use softmax activation funtion 
        probabilities = F.softmax(self.actor.forward(observation))
        # we then create a distribution that is modelled on the probabilities 
        action_probs = T.distributions.Categorical(probabilities)
        # then we get the action by sampling the action probability space 
        action = action_probs.sample()
        # Now we need the log probability of our sample to perform back-prop
        self.log_probs = action_probs.log_prob(action)

        # then we want to return our action as a integer using .item() since thats what 
        # OpenAI uses (action is currently a cuda tesnor)
        return action.item()

    # next we create an interface between the agent and it's memory
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        # when we initiate the agent value = target_value (networks)
        # however on every other step we want it to be a soft copy, to do this
        # we use tau as a scaler
        if tau is None:
            tau = self.tau

        # here we make a copy of the parameters, modify them and then upload them
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*(value_state_dict[name].clone()) + \
                (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    # Below is the implementation of the learning functionality of our agent.
    def learn(self):
        # Step 1: check if memory batch is full, 
        # if it's full we're not going to learn (i.e. go back through main loop)
        if self.memory.mem_cntr < self.batch_size:
            return 

        # if it's full we want to sample our memory buffer and get arrays
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        # Step 2: we need to convert numpy arrays into pytorch tensors 

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Step 3: calculate the values of the states and new_states according to the
        # value and target_value networks respectively use view(-1) to return tensor w/ proper dimension
        value = self.value(state).view(-1) #(q_eval)
        value_ = self.target_value(state_).view(-1) #(q_next)
        value_[done] = 0.0

        # Step 4: Value network loss 
        # we need to get the actions and log probabilities for the states
        actions = self.actor.forward(state)
        # we need the q values from our critics we get this by passing our 
        # current state and action through the critic networks and use minimum (for improved stability) 
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - self.log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target) 
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Step 5: actor network loss. 
        # start doing another feed-forward to get actions 
        # here we do use reparameterization 
        actions = self.actor.forward(state)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = self.log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Step 6: Critic Loss 
        # this is similar to what you would do when using Q learning
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        # this is where the actions from our REPLAYBUFFER come in
        # we're using the action from the replay buffer here!!!
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()








        







        


