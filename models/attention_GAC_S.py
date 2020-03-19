import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time

import math


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, rpe=False, deg=False, scale=False, return_kv=False, layer=0):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.rpe = rpe
        self.scale = scale
        self.deg = deg
     
        self.return_kv = return_kv
        self.layer = layer

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        if self.rpe:
            self.rel_k = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True) # 32 x 1 x 7


        if self.deg:
            #self.query_conv = nn.Conv2d(in_channels//2+1, out_channels, kernel_size=1, bias=bias)
            #self.key_conv = nn.Conv2d(in_channels+1, out_channels, kernel_size=1, bias=bias)
            #self.value_conv = nn.Conv2d(in_channels+1, out_channels, kernel_size=1, bias=bias)
            self.query_conv = nn.Conv2d(in_channels//2+1, out_channels, kernel_size=1, bias=bias)
            self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        else:
            self.query_conv = nn.Conv2d(in_channels//2, out_channels, kernel_size=1, bias=bias)
            self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)


        #self.reset_parameters()

    def forward(self, x, abs_x, deg, idx):
        batch, channels, npoints, neighbors = x.size() # B, C, N, K

        ''' 1. get point features (B, C, N) '''
        x_q = abs_x # B, C//2, N, 1
        x_kv = x # B, C, N, K

        ''' 2. transform by Wq, Wk, Wv '''
        q_out = self.query_conv(x_q) # B, C, N, 1
        k_out = self.key_conv(x_kv) # B, C, N, K
        k_out_all = k_out[:,:,:,0] # B, C, N, 1
        v_out = self.value_conv(x_kv) # B, C, N, K
        v_out_all = v_out[:,:,:,0] # B, C, N, 1

        ''' 3. relative positional encoding ''' 
        if self.rpe:
            k_out = k_out + self.rel_k 
  
        # k_out : B, C, N, K / self.rel_k : C, 1, K

        ''' 4. multi-head attention '''
        if self.scale:
            scaler = torch.tensor([self.out_channels / self.groups]).cuda()
            out = torch.rsqrt(scaler) * q_out * k_out
        else:
            out = q_out * k_out # B, C, N, K

        out = F.softmax(out, dim=-1) # B, C, N, K

        ''' 5. scoring '''
        idx = idx[:,:,:,-1].unsqueeze(1).expand_as(out).cuda() # B, N, K -> B, 1, N, K -> B, C, N, K
        idx_scatter = torch.zeros(batch, self.out_channels, npoints, npoints, device='cuda').detach() # B, C, N, N

        # node-wise importance
        idx_scatter.scatter_(dim=3, index=idx, src=out)[0,0,0,:] # B,C,N,N -> B,C,N,1            
        #score = idx_scatter.sum(dim=2, keepdim=True).transpose(2,3).repeat(1,1,1,npoints) # B, C, 1, N -> B, C, N, 1 -> B, C, N, N

        #idx_key, idx_salient = score.topk(k=int(1024/16), dim=-1) # B, C, N, S | here, S : sampled global points


        score = idx_scatter.sum(dim=2, keepdim=True).transpose(2,3).squeeze(3) # B, C, 1, N -> B, C, N, 1 -> B, C, N 

        idx_key, idx_salient = score.topk(k=20, dim=-1) # B, C, S | here, S : sampled global points

        k_out_all = torch.gather(k_out_all, 2, idx_salient).unsqueeze(2).repeat(1,1,npoints,1) # B, C, S -> B, C, N, S
        v_out_all = torch.gather(v_out_all, 2, idx_salient).unsqueeze(2).repeat(1,1,npoints,1) # B, C, S -> B, C, N, S

        out_all = q_out * k_out_all # B, C, N, S
        out_all = F.softmax(out_all, dim=-1) # B, C, N, S 
        out_all = torch.einsum('bcns,bcns -> bcn', out_all, v_out_all) # B, C, N
        out_all = out_all.view(batch, -1, npoints, 1) # B, C, N, 1
      
        # x : 6 x 3  /  idx : 6
        # x : B,C,N x K / idx : B,C,N,K

        out = torch.einsum('bcnk,bcnk -> bcn', out, v_out) # b, C, N, K
        out = out.view(batch, -1, npoints, 1) # b, C, N, 1


        ''' 8. reshape for memory '''
        #out = torch.cat([out, out_all], dim = 1)
        if self.layer > 1:
            #print("ratio : {}".format(out_all.mean()/out.mean()))

            out = out + out_all 


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

        if self.rpe:
            init.normal_(self.rel_k, 0, 1)
        #init.normal_(self.rel_w, 0, 1)





class MemoryAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, rpe=False, scale=False, layer=1, deg=False):
        super(MemoryAttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.rpe = rpe
        self.layer= layer
        self.deg = deg

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        if self.rpe:
            self.rel_k = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True) # 32 x 1 x 7

        if self.deg:
            self.query_conv = nn.Conv2d(in_channels//2+1, out_channels, kernel_size=1, bias=bias)
            self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        else:
            self.query_conv = nn.Conv2d(in_channels//2, out_channels, kernel_size=1, bias=bias)
            self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()


    def forward(self, x, k, v, s, abs_x, deg, idx):
        batch, channels, npoints, neighbors = x.size() # B, C, N, K

        ''' 1. get point features (B, C, N) '''
        x_q = abs_x # B, C, N, 1 
        x_kv = x # B, C, N, K

        ''' 2. Query : transform x to q, k, v by Wq, Wk, Wv '''
        q_query = self.query_conv(x_q) # B, C, N, 1
        k_query = self.key_conv(x_kv) # B, C, N, K
        v_query = self.value_conv(x_kv) # B, C, N, K
        #print(idx.shape) # B, N, K, T
        idx_query = idx[:,:,:,-1].unsqueeze(1).expand(-1, self.out_channels,-1, -1) # B, N, K, T -> B, N, K -> B, C, N, K
        #print(idx_query.shape) # B, G, N, K 

        ''' 3. Memory : retrieve previous k, v ''' 
        k_memory = k # B, C, N, K, T
        v_memory = v # B, C, N, K, T
        s_memory = s # B, C, N, T
        idx_memory = idx[:,:,:,:-1].unsqueeze(1).expand(-1, self.out_channels, -1, -1, -1) # B, C, N, K, T

        ''' 4. relative positional encoding ''' 
        # pos-encoding by knn orders.
        if self.rpe:
            k_query = k_query + self.rel_k 
        # k_out : B, C, N, K / self.rel_k : C, 1, K

        ''' 5. multi-head attention for Query '''
        #q_query = q_query.view(batch, self.groups, self.out_channels // self.groups, npoints, 1) # B, G, C//G, N, 1
        #k_query = k_query.contiguous().view(batch, self.groups, self.out_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        #v_query = v_query.contiguous().view(batch, self.groups, self.out_channels // self.groups, npoints, -1) # b, G, C//G, N, K

        out_query = q_query * k_query # B, C, N, K
        out_query = F.softmax(out_query, dim=-1) # B, C, N, K

        ''' 6. scoring current layer(Query) '''
        idx_scatter_query = torch.zeros(batch, self.out_channels, npoints, npoints, device='cuda').detach() # B, C, N, N


        # node-wise importance
        idx_scatter_query.scatter_(dim=3, index=idx_query, src=out_query) # B,C,N,N -> B,C,N,1
        score_query = idx_scatter_query.sum(dim=2, keepdim=True).transpose(2,3)

        #out_query = out_query.unsqueeze(2).expand_as(v_query) # B, C, N, 1

        out_query = torch.einsum('bcnk,bcnk -> bcn', out_query, v_query) # B, C, N
        out_query = out_query.view(batch, -1, npoints, 1) # b, C, N, 1


        ''' 7. multi-head attention for Memory '''
        k_memory = k_memory.contiguous().view(batch, self.out_channels, npoints, neighbors*self.layer) # B, C, N, KT

        v_memory = v_memory.contiguous().view(batch, self.out_channels, npoints, neighbors*self.layer) # B, C, N, KT


        ''' 8. retrieve salient features from k_memory/v_memory using s_memory '''
        # k/v_memory : B, C, N, KT
        # s_memory : B, C, N, T

        s_memory = s_memory.permute(0, 1, 3, 2) # B, C, N, T -> B, C, T, N
        s_memory = s_memory.unsqueeze(4).expand(-1, -1, -1, -1, npoints) # B, C, T, N, N

        idx_memory = idx_memory.permute(0, 1, 4, 2, 3) # B, C, N, K, T -> B, C, T, N, K

        s_memory = torch.gather(s_memory.transpose(3,4), 4, idx_memory) # B, C, T, N, K

        s_memory = s_memory.permute(0, 1, 3, 4, 2) # B, C, N, K, T
        s_memory = s_memory.contiguous().view(batch, self.out_channels, npoints, neighbors*self.layer) # B, C, N, KT

        idx_key, idx_salient = s_memory.topk(k=5*self.layer, dim=-1) # B, C, N, K'T

        ''' check idx learned '''
        #print(idx_key[4,0,0:5,:])
        #print(idx_salient[4,0,0:5,:])

        ''' check idx learned end '''

        k_memory = torch.gather(k_memory, 3, idx_salient)
        v_memory = torch.gather(v_memory, 3, idx_salient)

        out_memory = q_query * k_memory # B, C, N, K'T

        out_memory = F.softmax(out_memory, dim=-1) # B, C, N, K'T

        out_memory = torch.einsum('bcnk,bcnk -> bcn', out_memory, v_memory) # B, C, N


        ''' 8. reshape for memory '''
        out_memory = out_memory.view(batch, -1, npoints, 1) # b, C, N, 1

        k_query = k_query.view(batch, -1, npoints, neighbors, 1)
        v_query = v_query.view(batch, -1, npoints, neighbors, 1)

        out = torch.cat([out_query, out_memory], dim = 1)

        return out, k_query, v_query, score_query


    def reset_parameters(self):
        #init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        #init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        #init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        #if self.rpe:
        #    init.normal_(self.rel_k, 0, 1)
        #init.normal_(self.rel_w, 0, 1)

        init.kaiming_normal_(self.key_conv.weight)
        init.kaiming_normal_(self.value_conv.weight)
        init.kaiming_normal_(self.query_conv.weight)

        if self.rpe:
            init.normal_(self.rel_k, 0, 1)
        #init.normal_(self.rel_w, 0, 1)

