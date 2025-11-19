import  torch
import torch.nn as nn
import numpy as np

class ChannelAttention(nn.Module): 
    """ Channel Attention Module :
        Focuses on 'what' is important given the feature maps.
    """
    
    def __init__(self , input_channels,reduction_ratio):
        pass 