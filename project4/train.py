import gymnasium as gym
import sys
import os
import torch
from pathlib import Path
import ale_py

# --- 1. Setup Game ROMs ---
# Register Atari environments
gym.register_envs(ale_py)
# Fix library conflict on Mac devices
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- 2. Import Custom Classes ---
from breakout import DQNBreakout
from dqn_agent import Agent

if __name__ == "__main__":
    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING TRAINING ON {device} ---")
    
    # Setup Environment
    # render_mode='rgb_array': Fast mode (no window)
    # render_mode='human': Slow mode (opens window to watch training)
    env = DQNBreakout(render_mode='rgb_array', device=device)
    
    # Create Agent (containing the Attention network)
    agent = Agent(env, device)
    
    # Number of episodes (increase for better results)
    episodes = 500
    
    try:
        for e in range(episodes):
            # Reset game for a new episode
            state, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # 1. Agent decides action (based on Epsilon)
                action = agent.act(state)
                
                # 2. Execute action in the environment
                next_state, reward, done, truncated, info = env.step(action)
                
                # 3. Store experience in memory
                agent.memory.push(state, action, reward, next_state, done)
                
                # 4. Train the network (Crucial step)
                agent.learn()
                
                # Update current state
                state = next_state
                total_reward += reward
                
                if done or truncated:
                    print(f"Episode {e+1}/{episodes} - Score: {total_reward} - Epsilon: {agent.epsilon:.2f}")
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining Paused by User.")
    finally:
        env.close()
        # Save final model
        save_path = "breakout_attention_model.pth"
        torch.save(agent.policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")