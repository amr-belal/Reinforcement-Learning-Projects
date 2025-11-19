import collections
import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array',repeat=4 , device='cpu'):
        # env = gym.make("AdventureNoFrameskip-v4",frameskip = 4, render_mode=render_node)
        # env = gym.make("AdventureDeterministic-v4" ,frameskip=4 , render_mode=render_mode)
        # env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        env = gym.make("ALE/Adventure-v5", render_mode=render_mode)
        super(DQNBreakout, self).__init__(env)
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.device = device
        self.frame_buffer = collections.deque(maxlen=2)
        self.render_mode = render_mode


    
    def step(self , action):
        total_reward = 0.0
        done = False

        for i in range(self.repeat):
            observation , reward,done , truncated, info = self.env.step(action)
            total_reward += reward

            print(info)


            # TODO : Decrement lives 
            self.frame_buffer.append(observation)

            if done :
                break
        
        max_frames = np.max(self.frame_buffer[-2:], axis=0)
        max_frames = torch.tensor(max_frames, device=self.device, dtype=torch.float32)
        max_frames = max_frames.to(self.device)

        if self.render_mode == 'human':
            self.render()



        return max_frames,total_reward,done,info

    
    def reset(self):
        obs, info = self.env.reset()
        self.frame_buffer.clear()
        self.frame_buffer.append(obs)
        if self.render_mode == 'human':
            self.render()
        return obs, info