r""" Enhanced Attention Mechanisms for BCHNet """

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelAttention(nn.Module):
    """Dual-path Spatial-Channel Attention Module"""
    
    def __init__(self, in_channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.spatial_conv(spatial_input)
        x_spatial = x_channel * spatial_att
        
        return x_spatial


class CrossAttention(nn.Module):
    """Cross Attention between Query and Support Features"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, query, support):
        B, C, H, W = query.shape
        
        # Reshape to sequence
        query_seq = query.flatten(2).transpose(1, 2)  # [B, HW, C]
        support_seq = support.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Project to Q, K, V
        q = self.q_proj(query_seq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(support_seq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(support_seq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class AdaptivePrototypeRefinement(nn.Module):
    """Adaptive Prototype Refinement with Multi-scale Context"""
    
    def __init__(self, in_channels):
        super(AdaptivePrototypeRefinement, self).__init__()
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feature, prototype):
        """
        feature: [B, C, H, W]
        prototype: [B, C, 1, 1]
        """
        # Expand prototype to feature size
        proto_expanded = prototype.expand_as(feature)
        
        # Concatenate and refine
        combined = torch.cat([feature, proto_expanded], dim=1)
        refined = self.refine_conv(combined)
        
        # Gated fusion
        gate = self.gate(refined)
        output = feature * gate + refined * (1 - gate)
        
        return output


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module for Multi-scale Context"""
    
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        pyramid_feats = []
        for conv in self.convs:
            pooled = conv(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            pyramid_feats.append(upsampled)
        
        pyramid_feats = torch.cat(pyramid_feats, dim=1)
        output = self.fusion(torch.cat([x, pyramid_feats], dim=1))
        
        return output


class FeatureFusionModule(nn.Module):
    """Enhanced Feature Fusion with Attention"""
    
    def __init__(self, in_channels_list):
        super(FeatureFusionModule, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, in_channels_list[0], 1) 
            for in_ch in in_channels_list
        ])
        
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_list[0], in_channels_list[0], 3, padding=1),
                nn.BatchNorm2d(in_channels_list[0]),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list) - 1)
        ])
        
        self.attention = SpatialChannelAttention(in_channels_list[0])
        
    def forward(self, features):
        """
        features: list of features from different levels
        """
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway with fusion
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=(h, w), mode='bilinear', align_corners=True
            )
            laterals[i-1] = self.fusion_convs[i-1](laterals[i-1])
        
        # Apply attention to final output
        output = self.attention(laterals[0])
        
        return output
