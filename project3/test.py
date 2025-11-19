import gymnasium as gym
import torch
import ale_py
import os
from breakout import DQNBreakout
from dqn_agent import Agent


gym.register_envs(ale_py)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def watch_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- WATCHING TRAINED AGENT ON {device} ---")

    
    env = DQNBreakout(render_mode='human', device=device)
    
   
    agent = Agent(env, device)


   
    model_filename = "breakout_model.pth" 
    
    if os.path.exists(model_filename):
        print(f"Loading model from {model_filename}...")
        
        agent.policy_net.load_state_dict(torch.load(model_filename, map_location=device))
        agent.policy_net.eval()
    else:
        print(f"Error: Could not find {model_filename}. Did you finish training?")
        return

    agent.epsilon = 0.02 


    num_episodes = 5
    
    try:
        for e in range(num_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            
            print(f"Starting Game {e+1}...")
            
            while not done:
                # الـ Agent بياخد قرار بناءً على اللي اتعلمه
                action = agent.act(state)
                
                # تنفيذ الحركة
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # مفيش هنا agent.learn() ولا memory.push() لأننا بنختبر بس

                if done or truncated:
                    print(f"Game {e+1} Finished! Score: {total_reward}")
                    break
                    
    except KeyboardInterrupt:
        print("Watching stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    watch_agent()