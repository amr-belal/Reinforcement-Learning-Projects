
import numpy as np 

# Environment setup 
GRID_SIZE = 4
GOAL_STATE = (3,3)
ACTIONS = ['U', 'D', 'L', 'R']
DISCOUNT = 0.9
REWARD_STEP = -0.04
REWARD_GOAL = 1.0
WALLS = [(1, 1)]

# INITIALIZE VALUES AND POLICY 

V = np.zeros((GRID_SIZE , GRID_SIZE ))

policy = np.full((GRID_SIZE , GRID_SIZE ), ' ')

# helper: give next state given an action

def step(state , action): 
    x,y = state 
    if state ==GOAL_STATE: 
        return state , 0 
    if action =='U':
        x = max(x-1 ,0)
    elif action =='D':
        x  = min(x+1 , GRID_SIZE -1)
    elif action =='L':
        y = max(y-1 ,0)
    elif action =='R':
        y = min(y+1 , GRID_SIZE -1)
        
    if (x,y) in WALLS:
        x,y = state
        
    reward = REWARD_GOAL if (x,y) == GOAL_STATE else REWARD_STEP
    return (x,y) , reward


    
def value_iteration(threshold=1e-4):
    global V
    iteration = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                if s == GOAL_STATE or s in WALLS:
                    continue
                values = []
                for a in ACTIONS:
                    s_next, reward = step(s, a)
                    values.append(reward + DISCOUNT * V[s_next])
                new_V[s] = max(values)
                delta = max(delta, abs(V[s] - new_V[s]))
        V = new_V
        iteration += 1
        if delta < threshold:
            break
    print(f"Value Iteration converged in {iteration} iterations.")
    return V


def extract_policy():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = (i, j)
            if s == GOAL_STATE or s in WALLS:
                policy[s] = '•'
                continue
            values = []
            for a in ACTIONS:
                s_next, reward = step(s, a)
                values.append(reward + DISCOUNT * V[s_next])
            best_action = ACTIONS[np.argmax(values)]
            policy[s] = best_action
    return policy



V = value_iteration()
policy = extract_policy()

print("Final Value Function:")
print(np.round(V, 2))
print("\nOptimal Policy:")
print(policy)
