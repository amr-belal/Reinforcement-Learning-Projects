# import random 
# import numpy as np
# import gymnasium as gym


# # initialize the environment 
# env = gym.make("Taxi-v3")

# alpha = 0.9 # learning rate ==> indicate the rate of how the new information overrides the old information ranges for 0to1 
# gamma = 0.95 # discount factor ==> determine how the impact of the future rewards within the immediate or current reward 
# epsilon = 1.0   # controls of the ransdomness or the exploration rate 
# epsilon_decay = 0.995 # to dcay of make the epsilon smaller over time 
# min_epsilon = 0.01
# num_episodes = 10000
# max_steps = 100
# # for taxi is 5x5 grid  -> 25 position *5 *4 ==> 500 different states 4 for four directions
# q_table = np.zeros((env.observation_space.n , env.action_space.n)) # how many states agent can be in x haw many actions agents made 


# def choose_action(state):
#     if random.uniform(0,1)<epsilon:
#         return env.action_space.sample()
#     else :
#         return np.argmax(q_table[state, :])


# for episode in range(num_episodes):
#     state ,_ =env.reset()

#     done = False

#     for step in range(max_steps):
#         action = choose_action(state)

#         next_state , reward , done , truncated ,info  = env.step(action)

#         # Q-learning update rule (fixed missing multiplication)
#         old_value = q_table[state, action]
#         next_max = np.max(q_table[next_state, :])
#         q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

#         state = next_state

#         if done or truncated :
#             break

#     epsilon = max(min_epsilon ,epsilon*epsilon_decay)


# env =gym.make("Taxi-v3",render_mode ="human" )


# for episode in range(5):
#     state,_ =env.reset()
#     done = False 
    

#     print("episodes :" ,episode)

#     for step in range(max_steps):
#         env.render()
#         action = np.argmax(q_table[state , :])

#         next_state ,reward , done , truncated ,info =  env.step(action)

#         state = next_state 
#         if done or truncated:
#             env.render()

#             print("fininshed  episode " , episode , " with reward ",reward)
#             break

# env.close()        


import random
import numpy as np
import gymnasium as gym

# Initialize the environment
env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()   # <-- fixed parentheses
    else:
        return np.argmax(q_table[state, :])

# Q-learning algorithm
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Q-learning update rule (fixed missing multiplication)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done:
            break

    # Decay exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Test the trained agent with rendering
env = gym.make("Taxi-v3", render_mode="human")

for episode in range(5):
    state, info = env.reset()
    terminated = False
    truncated = False
    print("Episode:", episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        if terminated or truncated:
            env.render()
            print("Finished episode", episode, "with reward", reward)
            break

env.close()
