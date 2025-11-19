import gymnasium as gym
import sys
import os
import torch
from pathlib import Path
import ale_py

# --- Setup ROMs ---
gym.register_envs(ale_py)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Imports ---
from breakout import DQNBreakout
from dqn_agent import Agent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING TRAINING ON {device} ---")
    
    # Use 'human' to watch, or None to train fast
    env = DQNBreakout(render_mode=None, device=device) # human if you want to watch
    agent = Agent(env, device)
    
    episodes = 500
    
    try:
        for e in range(episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # 1. Act
                action = agent.act(state)
                
                # 2. Step
                next_state, reward, done, truncated, info = env.step(action)
                
                # 3. Store
                agent.memory.push(state, action, reward, next_state, done)
                
                # 4. Learn
                agent.learn()
                
                state = next_state
                total_reward += reward
                
                if done or truncated:
                    print(f"Episode {e+1} - Score: {total_reward} - Epsilon: {agent.epsilon:.2f}")
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining Paused.")
    finally:
        env.close()
        torch.save(agent.policy_net.state_dict(), "breakout_model.pth")
        print("Model saved.")