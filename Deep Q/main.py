import gym
from agent import Agent
from utils import plot_learning_curve
import numpy as np
from buffer import MemoryBuffer

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_end=0.01, input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done: 
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            scores.append(score)
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])

        print(f'Episode: {i}, Score: {score}, Avg Score: {avg_score}, Epsilon: {agent.epsilon}')

    x = [i+1 for i in range(n_games)] # create x-axis for graph
    filename = 'deepQ_Lander.png'
    plot_learning_curve(x, scores, eps_history, filename)

