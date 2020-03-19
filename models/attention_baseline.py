import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, rpe=False, scale=False, return_kv=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.rpe = rpe
        self.scale = scale

        self.return_kv = return_kv

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        if self.rpe:
            self.rel_k = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True) # 32 x 1 x 7

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels//2, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)


    def forward(self, x, abs_x):
        batch, channels, npoints, neighbors = x.size() # B, C, N, K

        ''' 1. get point features (B, C, N) '''
        x_q = abs_x # B, C, N, 1 
        x_kv = x # B, C, N, K

        ''' 2. transform by Wq, Wk, Wv '''
        q_out = self.query_conv(x_q) # B, C, N, 1
        k_out = self.key_conv(x_kv) # B, C, N, K
        v_out = self.value_conv(x_kv) # B, C, N, K

        ''' 3. relative positional encoding ''' 
        if self.rpe:
            k_out = k_out + self.rel_k 
  
        # k_out : B, C, N, K / self.rel_k : C, 1, K

        ''' 4. multi-head attention '''
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, npoints, -1) # b, groups, C//groups, N, K
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, npoints, -1) # b, groups, C//groups, N, K
        q_out = q_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, npoints, 1) # b, groups, C//groups, N, 1

        if self.scale:
            scaler = torch.tensor([self.out_channels / self.groups]).cuda()
            out = torch.rsqrt(scaler) * q_out * k_out
        else:
            out = q_out * k_out # b, groups, C//groups, N, K
        out = F.softmax(out, dim=-1) # b, groups, C//groups, N, K

        out = torch.einsum('bgcnk,bgcnk -> bgcn', out, v_out) # b, groups, C//groups, N
        out = out.view(batch, -1, npoints, 1) # b, C, N, 1

        if self.return_kv:
            k_out = k_out.view(batch, -1, npoints, neighbors, 1)
            v_out = v_out.view(batch, -1, npoints, neighbors, 1)
            
            return out, k_out, v_out
        else:
            return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')









