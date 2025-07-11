import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import re
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange
import os
import urllib.request
from tqdm import tqdm
import math
from model.wavelettransform import EnhancedWaveletTransformer
from model.convnext import convnext_tiny



model_urls = {
    "convnext_tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"
}

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        self.q1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q3_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.sig = nn.Sigmoid()
        
        self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, frame1_feat, center_feat, frame3_feat):
        B, C, H, W = center_feat.shape
        
        f1_pool = self.gap(frame1_feat).view(B, C)  # [B, C]
        c_pool = self.gap(center_feat).view(B, C)   # [B, C]
        f3_pool = self.gap(frame3_feat).view(B, C)  # [B, C]
        
        q1 = self.q1_proj(f1_pool)  # [B, C]
        q3 = self.q3_proj(f3_pool)  # [B, C]
        k = self.k_proj(c_pool)     # [B, C]
        v = self.v_proj(c_pool)     # [B, C]
        
        q1_expanded = q1.unsqueeze(2)  # [B, C, 1]
        k_expanded = k.unsqueeze(1)    # [B, 1, C]
        similarity1 = torch.bmm(q1_expanded, k_expanded) * self.scale  # [B, C, C]
        attn1 = F.softmax(similarity1, dim=2)  
        attn1 = self.dropout(attn1)
        
        q3_expanded = q3.unsqueeze(2)  # [B, C, 1]
        similarity3 = torch.bmm(q3_expanded, k_expanded) * self.scale  # [B, C, C]
        attn3 = F.softmax(similarity3, dim=2)  
        attn3 = self.dropout(attn3)
        
        attn1_col_sum = attn1.sum(dim=1)  
        attn3_col_sum = attn3.sum(dim=1)  
        
        channel_weights1 = attn1_col_sum * v  
        channel_weights3 = attn3_col_sum * v  
        
        weight1 = self.sig(channel_weights1).view(B, C, 1, 1)  # [B, C, 1, 1]
        weight3 = self.sig(channel_weights3).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        fusion_weights = F.softmax(self.fusion_weights, dim=0)  # [2]
        combined_weight = weight1 * fusion_weights[0] + weight3 * fusion_weights[1]  # [B, C, 1, 1]
        
        enhanced_feat = center_feat * combined_weight  # [B, C, H, W]
        
        return enhanced_feat


class CrossAttention(nn.Module):
    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        
        self.conv_reduce = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        
        self.weight = nn.Parameter(torch.Tensor(1, 1, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1))
        
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def _generate_attention(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        pooled_feat = torch.cat([max_pool, avg_pool], dim=1)  # [B, 2, H, W]
        
        pooled_feat = self.conv_reduce(pooled_feat)  # [B, 1, H, W]
        
        attention = pooled_feat * self.weight + self.bias
        
        attention = torch.sigmoid(attention)
        
        attention = self.dropout(attention)
        
        return attention
    
    def forward(self, feat1, feat2):
        
        attn = self._generate_attention(feat2)  # [B, 1, H, W]
        
        enhanced_feat1 = feat1 * attn  
        
        return enhanced_feat1


class HCMI(nn.Module):
    def __init__(self, in_chans=3, out_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.,
                 load_pretrained=True, hidden_dim=384, num_heads=8):
        super().__init__()
        
        self.wavelet_transformer = EnhancedWaveletTransformer(in_channels=in_chans, hidden_dim=hidden_dim)
        
        self.backbone_type = "convnext"
        
        print(f"创建ConvNeXt-Tiny模型作为主干网络，{'加载' if load_pretrained else '不加载'}预训练权重")
        
        self.backbone = convnext_tiny(
            pretrained=False,  
            in_chans=in_chans
        )
        
        if load_pretrained:
            try:
                model_path = model_urls['convnext_tiny']
                print(f"从URL {model_path} 加载ConvNeXt预训练权重")
                
                checkpoint = torch.hub.load_state_dict_from_url(url=model_path, map_location="cpu", check_hash=True)
                
                if "model" in checkpoint:
                    pretrained_dict = checkpoint["model"]
                else:
                    pretrained_dict = checkpoint
                
                new_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    if k.startswith('backbone.'):
                        k = k[9:]  
                    if k.startswith('module.'):
                        k = k[7:]  
                    if not k.startswith('head.'):
                        new_pretrained_dict[k] = v
                
                model_dict = self.backbone.state_dict()
                
                pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
                
                res = self.backbone.load_state_dict(pretrained_dict, strict=False)
                print(f"成功加载ConvNeXt预训练权重")
                print(f"缺少的键: {len(res.missing_keys)}, 意外的键: {len(res.unexpected_keys)}")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
                import traceback
                traceback.print_exc()
        
        self.channel_attn1 = ChannelAttention(
            dim=dims[3],  
            dropout=0.1
        )
        
        self.channel_attn2 = ChannelAttention(
            dim=dims[2],  
            dropout=0.1
        )
        
        self.cross_attn1 = CrossAttention(
            feat_dim=dims[3],  
            dropout=0.1
        )

        self.cross_attn2 = CrossAttention(
            feat_dim=dims[2],  
            dropout=0.1
        )
        
        self.upsample1_f = nn.Sequential(
            nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2),
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        self.upsample1_s = nn.Sequential(
            nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2),
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        self.upsample2_f = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        self.upsample2_s = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            LayerNorm(dims[1] * 2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1] * 2, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.density_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),  
            nn.Conv2d(32, out_chans, kernel_size=1)
        )
        
        nn.init.kaiming_normal_(self.density_head[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.density_head[0].bias, 0)
        
        nn.init.constant_(self.density_head[-1].bias, 0.05)  
        nn.init.normal_(self.density_head[-1].weight, mean=0.01, std=0.005)  

    
    def extract_features(self, x):
        B = x.shape[0]
        
        features = []
        
        x = self.backbone.downsample_layers[0](x)
        x = self.backbone.stages[0](x)
        features.append(x)  
        
        x = self.backbone.downsample_layers[1](x)
        x = self.backbone.stages[1](x)
        features.append(x)  
        
        x = self.backbone.downsample_layers[2](x)
        x = self.backbone.stages[2](x)
        features.append(x)  
        
        x = self.backbone.downsample_layers[3](x)
        x = self.backbone.stages[3](x)
        features.append(x)  
        
        return features
    
    def forward(self, x, ego_motion=None):
        
        B, T, C, H, W = x.shape
        assert T == 3, "输入必须包含三帧图像"
        
        frame1 = x[:, 0]  
        center_frame = x[:, 1]  
        frame3 = x[:, 2]  
        
        center_features = self.extract_features(center_frame)
        
        lowfreq_features = self.wavelet_transformer(x)  # [B, T, C, H/2, W/2]
        
        upsampled_features = []
        for t in range(T):
            current_lowfreq = lowfreq_features[:, t]
            
            current_upsampled = F.interpolate(
                current_lowfreq, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            
            upsampled_features.append(current_upsampled)
        
        lowfreq1_features = self.extract_features(upsampled_features[0])
        lowfreq_center_features = self.extract_features(upsampled_features[1])
        lowfreq3_features = self.extract_features(upsampled_features[2])
        
        center_last = center_features[-1]     
        
        s1_last = lowfreq1_features[-1]       
        s2_last = lowfreq_center_features[-1] 
        s3_last = lowfreq3_features[-1]       
        
        enhanced_s2 = self.channel_attn1(s1_last, s2_last, s3_last)  # [B, C, H/32, W/32]
        
        enhanced_f1 = self.cross_attn1(center_last, enhanced_s2)  # [B, C, H/32, W/32]
        
        enhanced_f1 = enhanced_f1 + center_last
        
        enhancedd_s2 = self.cross_attn1(enhanced_s2, center_last)  # [B, C, H/32, W/32]
        
        enhanceds2 = enhancedd_s2 + enhanced_s2
        
        f2 = self.upsample1_f(enhanced_f1)  # [B, C/2, H/16, W/16]
        s2 = self.upsample1_s(enhanceds2)  # [B, C/2, H/16, W/16]
        
        s1_stage3 = lowfreq1_features[-2]       
        s3_stage3 = lowfreq3_features[-2]       
        
        enhanced_s2_stage2 = self.channel_attn2(s1_stage3, s2, s3_stage3)  # [B, C/2, H/16, W/16]
        
        enhanced_f2 = self.cross_attn2(f2, enhanced_s2_stage2)  # [B, C/2, H/16, W/16]
        
        enhanced_f2 = enhanced_f2 + f2
        
        enhanced_s2_final = self.cross_attn2(enhanced_s2_stage2, f2)  # [B, C/2, H/16, W/16]
        
        enhanced_s2_final = enhanced_s2_final + s2
        
        f3 = self.upsample2_f(enhanced_f2)  # [B, C/4, H/8, W/8]
        s3 = self.upsample2_s(enhanced_s2_final)  # [B, C/4, H/8, W/8]
        
        fused_feat = torch.cat([f3, s3], dim=1)  # [B, C/2, H/8, W/8]
        
        fused_feat = self.fusion(fused_feat)  # [B, 64, H/8, W/8]
        
        density = self.density_head(fused_feat)  # [B, 1, H/8, W/8]
        
        return density


if __name__ == '__main__':
    model = HCMI(
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        load_pretrained=True
    )
    
    batch_size = 1
    frames = 3  
    channels = 3  
    height = 128
    width = 128
    
    x = torch.rand(batch_size, frames, channels, height, width)
    
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

