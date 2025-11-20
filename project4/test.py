import gymnasium as gym
import torch
import ale_py
import os
from breakout import DQNBreakout
from dqn_agent import Agent

# Register Environment
gym.register_envs(ale_py)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def watch_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- WATCHING TRAINED AGENT ON {device} ---")

    # 1. Setup Environment in Watch Mode
    # render_mode='human' is necessary here to watch the game
    env = DQNBreakout(render_mode='human', device=device)
    
    # 2. Setup Agent
    agent = Agent(env, device)

    # 3. Load Brain (Trained Model)
    model_filename = "breakout_attention_model.pth" 
    
    if os.path.exists(model_filename):
        print(f"Loading model from {model_filename}...")
        # map_location ensures model loads even if trained on a different device
        agent.policy_net.load_state_dict(torch.load(model_filename, map_location=device))
        
        # Enable eval mode (stops Dropout and Batch Norm if present)
        agent.policy_net.eval() 
    else:
        print(f"Error: Could not find {model_filename}. Please run train.py first!")
        return

    # 4. Disable randomness (Serious play)
    # Set very small epsilon (0.02) to avoid stuck loops, mainly relying on intelligence
    agent.epsilon = 0.02 

    # Number of games to watch
    num_episodes = 5
    
    try:
        for e in range(num_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            
            print(f"Starting Game {e+1}...")
            
            while not done:
                # Agent picks the best action
                action = agent.act(state)
                
                # Execute action
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Note: No agent.learn() here because we are only testing

                if done or truncated:
                    print(f"Game {e+1} Finished! Score: {total_reward}")
                    break
                    
    except KeyboardInterrupt:
        print("Watching stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    watch_agent()