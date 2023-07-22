import torch
import math
import logging
import torchvision.models

import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        #print('mask',mask.shape)
        #print('attn',attn.shape)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x),mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class MoPRL(nn.Module):
    def __init__(self, tracklet_len=8, pre_len=1,headless=False, in_chans=2,embed_dim=128, spatial_depth=4, temporal_depth=4, 
                 num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  norm_layer=None, num_joints=17,
                 ):

        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            headless (bool): use head joints or not
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        # if headless:
        #     self.num_joints = 14
        # else:
        #     self.num_joints = 17

        self.num_joints = num_joints
        self.tracklet_len = tracklet_len
        self.pre_len = pre_len

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim 
        out_dim = 2

        ### spatial patch embedding
        self.pose_embedding = nn.Linear(in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr_s = [x.item() for x in torch.linspace(0, drop_path_rate, spatial_depth)]  # stochastic depth decay rule
        self.spatial_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_s[i], norm_layer=norm_layer)
                for i in range(spatial_depth)])
        self.spatial_norm = norm_layer(embed_dim)

        self.spatial_position_embedding = nn.Embedding(self.num_joints+1,embed_dim)

        dpr_t = [x.item() for x in torch.linspace(0, drop_path_rate, temporal_depth)]  # stochastic depth decay rule
        self.temporal_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_t[i], norm_layer=norm_layer)
                for i in range(temporal_depth)])
        self.temporal_norm = norm_layer(embed_dim)

        self.temporal_postion_embedding = nn.Embedding(tracklet_len+1,embed_dim)

        self.head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim , out_dim),
                    )
        
    def Spatial_Attention(self, x, spatial_tokens):

        b,f,_,_ = x.shape
        x = rearrange(x, 'b f w c -> (b f) w c',)

        spatial_embedding = self.spatial_position_embedding(spatial_tokens)
        _,_,n,d= spatial_embedding.shape
        spatial_embedding = spatial_embedding.view(-1,n,d)
        x += spatial_embedding

        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f w c', f=f)

        return x

    def Temporal_Attention(self, x, temporal_tokens):
        # x: b, T, N, embed_dim
        temporal_tembedding = self.temporal_postion_embedding(temporal_tokens)
        x += temporal_tembedding

        features = self.pos_drop(x)

        _, t, n, _ = features.shape
        features = rearrange(features, 'b t n c -> (b n) t c', t=t)

        for blk in self.temporal_blocks:
            features = blk(features)

        features = self.temporal_norm(features)

        features = rearrange(features, '(b n) t c -> b t n c', n=n)

        return features


    def forward(self, pose, spatial_tokens, temporal_tokens):
        # pose: (B, T, 17, 2)
        # box: (B, T, 2, 2)
        # spatial_embedding: (B, T, 17)
        pose = pose.permute(0, 3, 1, 2)
        b, _, f, p = pose.shape  ##### b is batch size, f is number of frames, p is number of joints
        pose = rearrange(pose, 'b c f p  -> (b f) p c', )
        pose = self.pose_embedding(pose)
        pose = rearrange(pose, '(b f) p c  -> b f p c', b=b)
        pose = self.Spatial_Attention(pose,spatial_tokens)
        pose = self.Temporal_Attention(pose,temporal_tokens) 
        rec_pose = self.head(pose).reshape(b,-1,2)

        return rec_pose

if __name__ == '__main__':
    model = MoPRL(tracklet_len=8, pre_len=1,headless=False, in_chans=2,embed_dim=64, spatial_depth=4, temporal_depth=4, 
                 num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  norm_layer=None, num_joints=25)

    pose = torch.rand([4,8,25,2])
    box = torch.rand([4,8,2,2])
    spatial_tokens = torch.randint(0,3,(4,8,25))
    temporal_tokens = torch.randint(0,3,(4,8,25))

    output = model(pose,spatial_tokens,temporal_tokens)

    print('output',output.shape)



