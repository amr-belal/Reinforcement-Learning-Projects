# import collections
# import gymnasium as gym
# import numpy as np
# import torch

# class DQNBreakout(gym.Wrapper):
#     def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
#         # Initialize the Breakout environment
#         env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
#         super(DQNBreakout, self).__init__(env)
        
#         self.repeat = repeat
#         # Use .unwrapped to access the emulator directly to fix 'OrderEnforcing' error
#         self.lives = env.unwrapped.ale.lives()
#         self.device = device
#         self.frame_buffer = collections.deque(maxlen=2)
        
#         # REMOVED: self.render_mode = render_mode 
#         # Reason: gym.Wrapper already has this property, setting it causes the error.

#     def step(self, action):
#         total_reward = 0.0
#         done = False
#         truncated = False
#         info = {}

#         # Repeat the action for 'repeat' frames (standard Atari practice)
#         for _ in range(self.repeat):
#             observation, reward, done, truncated, info = self.env.step(action)
#             total_reward += reward

#             # Store observation in buffer for max-pooling
#             self.frame_buffer.append(observation)

#             if done or truncated:
#                 break
        
#         # --- PROCESS FRAMES ---
#         # 1. Max-pool across the last 2 frames (removes flickering)
#         # Note: We convert deque to list first to avoid indexing errors
#         max_frame = np.max(np.stack(list(self.frame_buffer)), axis=0)
        
#         # 2. Normalize pixel values (0-255 -> 0.0-1.0)
#         max_frame = max_frame.astype(np.float32) / 255.0
        
#         # 3. Convert to PyTorch Tensor
#         max_frame_tensor = torch.tensor(max_frame, device=self.device, dtype=torch.float32)
        
#         # (Optional) If your CNN expects Channels First (C, H, W), uncomment below:
#         # max_frame_tensor = max_frame_tensor.permute(2, 0, 1)

#         # self.render_mode is accessed via the wrapper property (read-only)
#         if self.render_mode == 'human':
#             self.render()

#         return max_frame_tensor, total_reward, done, truncated, info

#     def reset(self):
#         # Handle the Gym reset
#         obs, info = self.env.reset()
        
#         self.frame_buffer.clear()
#         self.frame_buffer.append(obs)
#         self.frame_buffer.append(obs) # Fill buffer to prevent empty errors
        
#         if self.render_mode == 'human':
#             self.render()
            
#         # Convert initial observation to Tensor
#         obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32) / 255.0
#         # obs_tensor = obs_tensor.permute(2, 0, 1) # Uncomment if using channels-first
        
#         return obs_tensor, info


import collections
import gymnasium as gym
import numpy as np
import torch
import cv2

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        super(DQNBreakout, self).__init__(env)
        
        self.repeat = repeat
        self.device = device
        
        # Buffer for the 'repeat' frames (to handle flickering)
        self.frame_buffer = collections.deque(maxlen=2)
        
        # Buffer for the 'stack' of 4 processed frames (to see motion)
        self.stack_buffer = collections.deque(maxlen=4)
        
        self.lives = env.unwrapped.ale.lives()

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}

        # 1. Repeat Action (Frame Skipping)
        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            self.frame_buffer.append(obs)
            
            # Use 'ale.lives()' to detect life loss (optional but helps training)
            current_lives = self.env.unwrapped.ale.lives()
            if current_lives < self.lives:
                total_reward = -1.0 # Penalize losing a life
                self.lives = current_lives
                # done = True # Optional: Treat life loss as terminal for training stability

            if done or truncated:
                break
        
        # 2. Max Pooling (Remove flickering)
        max_frame = np.max(np.stack(list(self.frame_buffer)), axis=0)
        
        # 3. Process (Grayscale + Resize + Normalize)
        processed_frame = self.process_observation(max_frame)
        
        # 4. Add to Stack
        self.stack_buffer.append(processed_frame)
        
        # 5. Return Stack as Tensor (4, 84, 84)
        # If buffer isn't full (start of game), pad with copies of first frame
        while len(self.stack_buffer) < 4:
            self.stack_buffer.append(processed_frame)
            
        stacked_state = np.stack(list(self.stack_buffer), axis=0)
        stacked_tensor = torch.tensor(stacked_state, device=self.device, dtype=torch.float32)
        
        return stacked_tensor, total_reward, done, truncated, info

    def reset(self):
        self.frame_buffer.clear()
        self.stack_buffer.clear()
        self.lives = self.env.unwrapped.ale.lives()
        
        obs, info = self.env.reset()
        self.frame_buffer.append(obs)
        
        # Process first frame and fill stack
        processed_frame = self.process_observation(obs)
        for _ in range(4):
            self.stack_buffer.append(processed_frame)
            
        stacked_state = np.stack(list(self.stack_buffer), axis=0)
        stacked_tensor = torch.tensor(stacked_state, device=self.device, dtype=torch.float32)
        
        return stacked_tensor, info

    def process_observation(self, observation):
        # Convert to Grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84 (Standard for DQN)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalize (0-1)
        return resized / 255.0