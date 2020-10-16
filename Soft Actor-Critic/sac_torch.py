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
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], tau=0.005,
            env=None, gamma=0.99, n_actions=2, max_size=1000000, 
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma 
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        #get our networks
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims)
        self.target_value = ValueNetwork(beta, input_dims,name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        #turn the observation that we get into a cuda tensor
        state = T.Tensor([observation]).to(self.actor.device)
        # we get actions and logprobs but don't neeed ther logprobs so we use '_'
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # inorder to return a cuda tensor to the cpu we need to convert it to cpu
        # and detatch it from the graph and turn it into a numpy array and take the
        # 0 element
        return actions.cpu().detach().numpy()[0]

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
        # the reason this wasn't done in the buffer itself is so that the
        # buffer is framework agnostic allowing it to work with tensorflow and karas
        # we're setting it to the actor device here but it doesn't matter because
        # the devices for all the networks are the same. 
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Step 3: calculate the values of the states and new_states according to the
        # value and target_value networks respectively
        # we use view(-1) to return a view of the tensor with 1 less dimension
        # in this case it collapses it along the batch dimension because this 
        # will return a tensor of tensors which doesn't make sense for a tensor
        # of scalers (i.e. you don't want the inner scalers to be wrapped in brackets)
        # e.g. ({x}, {y}, {z}) = (x, y, z)
        # where the new states are terminal we want to set them to 0
        # this is just the definition of the value funtion 
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        # Step 4: Value network loss 
        # we need to get the actions and log probabilities for the states
        # according the the new policy (not for the actions that were actually
        # sampled from our buffer). This is because in the calculation for the 
        # loss of our value network and actor network we want the value of the 
        # actions according to our current policy!
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        # we need the q values from our critics we get this by passing our 
        # current state and action through the critic networks and use minimum
        # we do this because it improves the stability of learning. This has to
        # do with the overestimation bias which happens as a consequence of using 
        # a max over actions in the Qlearning update rule and as a consequence of
        # using deep NNs generally (as demonstrated in the TD3 paper). One way 
        # around this is to create a direct analog of the double-Qlearning update
        # rule by taking the values of the states actions with respect to 2 
        # different functions and then taking their minimum instead of the max
        # this is explained more in phils TD3 video 
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Next we calculate the value network loss and back propogate
        # calculate loss retaining the graph. we must retain the graph because
        # there is coupling between the losses between our networks. We need these 
        # graphs to calculate the losses for the actor and value networks
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target) #F.mse_loss = mean squared error loss
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Step 5: actor network loss. 
        # start doing another feed-forward to get actions and log-probs
        # here we do use reparameterization 
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q1_new_policy = self.critic_1.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
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








        







        


