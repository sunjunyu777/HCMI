import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn.init import trunc_normal_
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torchvision.utils as vutils
import math


class EnhancedWaveletTransformer(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        
        self.direction_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(3)
        ])
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dw_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
        
        self.transpose_conv = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        
    def _process_frame(self, frame):
        coeffs = self.dwt(frame)
        lowfreq = coeffs[0]  
        highfreq = coeffs[1][0]  
        
        B, C, D, Hf, Wf = highfreq.shape
        direction_feats = []
        
        for d in range(D):  
            direction_feat = highfreq[:, :, d, :, :]
            
            direction_feat = self.direction_convs[d](direction_feat)  # [B, 1, H/2, W/2]
            direction_feats.append(direction_feat)
        
        high_freq_concat = torch.cat(direction_feats, dim=1)
        
        pooled_feat = self.max_pool(high_freq_concat)  # [B, 3, H/4, W/4]
        
        dw_feat = self.dw_conv(pooled_feat)  # [B, 3, H/4, W/4]
        
        upsampled_feat = self.transpose_conv(dw_feat)  # [B, 3, H/2, W/2]
        
        attention_map = torch.sigmoid(upsampled_feat)  # [B, 3, H/2, W/2]
        
        enhanced_lowfreq = lowfreq * attention_map
        
        enhanced_lowfreq = enhanced_lowfreq + lowfreq
        
        return enhanced_lowfreq
    
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        assert T == 3, "输入必须是3帧图像"
        
        enhanced_lowfreqs = []
        
        for t in range(T):
            current_frame = frames[:, t, :, :, :]
            
            enhanced_lowfreq = self._process_frame(current_frame)
            
            enhanced_lowfreqs.append(enhanced_lowfreq)
        
        enhanced_frames = torch.stack(enhanced_lowfreqs, dim=1)
        
        return enhanced_frames




