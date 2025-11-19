import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections
import numpy as np

class ReplayBuffer:
    """
    applies experience replay buffer idea 
    stores experience tuples in a queue and samples from it randomly 
    
    append-only replay buffer for storing experience tuples.
    """
    def __init__(self,  capacity):
        """ capacity : maximum number of experience tuples to store in the buffer """
        self.buffer = collections.deque(maxlen=capacity) # deque is a double ended queue data structure that allows appending and popping from both ends with O(1) complexity
    
    def push(self ,state , action  ,reward ,  next_state , done):
        """ for adding experience tuples to the buffer """
        state  = state.cpu().numpy() if isinstance(state  , torch.Tensor) else state
        next_state  = next_state.cpu().numpy() if isinstance(next_state  , torch.Tensor) else next_state
        
        self.buffer.append((state , action , reward , next_state , done)) # store  the experience tuple in the buffer
        
    def sample(self, batch_size):
        """ randomly samples a batch of experience tuples from the buffer """
        
        batch = random.sample(self.buffer , batch_size)
        state , action , reward , next_state , done = zip(*batch)   
        
        return state , action , reward , next_state  , done        
    
    def __len__(self):
        """ returns the current size of the buffer """
        return len(self.buffer)