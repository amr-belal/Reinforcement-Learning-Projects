import torch
import torch.nn as nn
import numpy as np

class ChannelAttention(nn.Module): 
    """ Channel Attention Module :
        Focuses on 'what' is important given the feature maps.
    """
    
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels, bias=False)
        )
        # FIXED: Corrected typo 'sigmod' -> 'sigmoid'
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        
        out = avg_out + max_out
        
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
    

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module :
        Focuses on 'where' is an informative part, given the feature maps.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # FIXED: Corrected typo 'sigmod' -> 'sigmoid'
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)
    
    
class CBAM(nn.Module):
    """ Convolutional Block Attention Module :
        Combines Channel and Spatial Attention Modules.
    """
    
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class Network(nn.Module):
    """ Example Network using CBAM Module.
    """
    
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()
        
        self.conv = nn.Sequential(
            # FIXED: Added '32' (out_channels) which was missing
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.attention = CBAM(64)
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)
        x = self.attention(x)
        return int(np.prod(x.size()))
    
    
    def forward(self, x):
        # FIXED: Corrected typo 'feastures' -> 'features'
        features = self.conv(x)
        attended_features = self.attention(features)
        
        flat_features = attended_features.view(attended_features.size(0), -1)
        
        return self.fc(flat_features)