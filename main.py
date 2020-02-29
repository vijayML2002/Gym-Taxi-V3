#importing the lib
import gym
import numpy as np
import random

#creating the env
env = gym.make('Taxi-v3')

#defining the constant
epsilon = 0.96
gamma = 0.91
lr = 0.001
no_epi = 100000
no_step = 100
decay = 0.96

print('The number of observation space -- {}'.format(env.observation_space.n))
print('The number of action space -- {}'.format(env.action_space.n))

#creating the q-table
q_table = np.zeros((env.observation_space.n,env.action_space.n))

#defining the choose action
def choose_action(state):
    if np.random.rand()<epsilon:
        action = env.action_space.sample()
        exp = 0
    else:
        action = np.argmax(q_table[state,])
        exp =1
    return action,exp


#defining update the q_table
def update(state,action,state2,reward):
    q_table[state,action] = q_table[state,action] + lr*(reward + np.max(q_table[state2,]) - q_table[state,action])


#rendering the game
def render():
    state = env.reset()
    score = 0
    while True:
        env.render()
        action = np.argmax(q_table[state,])
        state2,reward,done,info = env.step(action)
        score+=reward
        if done:
            break
        state = state2
    print('The score of the game -- {}'.format(score))
    

#defining the game loop
for i in range(no_epi):
    state = env.reset()
    score = 0
    step = 0
    exploit = 0
    while step<no_step:
        action,exp = choose_action(state)
        state2,reward,done,info = env.step(action)
        update(state,action,state2,reward)
        if done:
            break
        if reward>0:
          reward+=100
        state =state2
        step+=1
        score+=reward
        exploit+=exp

    if i%10000==0:
        print('EPISODE - {}   SCORE - {}   EXPLOIT - {}'.format(i,score,exploit))

    if i%100==0:
      epsilon*=decay

    if epsilon<0.01:
      epsilon = 0.8


render()
