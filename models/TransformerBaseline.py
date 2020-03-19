#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .attention_baseline import AttentionConv


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx

def get_degree_distrib(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # 1, 1, 1

    idx = idx + idx_base # 1, 1024, 20

    _, num_dims, _ = x.size() # 3
    x = x.transpose(2, 1).contiguous() # 1, 1024, 3

    #########################################
    # degree distribution
    
    # idx : batch, npoints, k
    degree_distrib = torch.zeros(batch_size, 1024).cuda() # batch, npoints
    version = 1
    if version == 1:
        idx = idx.view(idx.shape[0], -1)
        for m in range(batch_size):
            degree_distrib[m] = torch.histc(idx[m], bins=1024, min=1024*(m), max=1024*(m+1)-1)

    #for m in range(batch_size):
    #    for n in range(num_points):
    #        degree_distrib[m, idx[m, n]] += 1

    #########################################

    x = torch.cat((x, degree_distrib.unsqueeze(2)), dim=2)

    return x


def get_neighbors(x, k=20, idx=None, deg=True, delta=True, neighbors=True, bn=None, ape=None, ptcld=None, static=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    abs_x = x.unsqueeze(3)

    if static is False:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    else:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        idx_initial = idx

    #########################################
    # absolute positional encoding
    if ape: 
        ptcld = ptcld.unsqueeze(3) # B, C, N -> B, C, N, 1
        ape_temp = ape(ptcld)
        ape_temp = ape_temp.squeeze(3)
        x = x + ape_temp # (B, C, N) + (B, 3, N) ==> (B, C, N)
    #########################################

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # 1, 1, 1

    idx = idx + idx_base # 1, 1024, 20

    _, num_dims, _ = x.size() # 3
    x = x.transpose(2, 1).contiguous() # 1, 1024, 3

    #########################################

    idx = idx.view(-1)

    feature = x.view(batch_size*num_points, -1)[idx, :] # 1024*20, 3
    feature = feature.view(batch_size, num_points, k, num_dims) # 1, 3, 1024, 20

    if delta and neighbors:
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # 1, 1024, 20, 3
        feature = torch.cat((feature-x, feature), dim=3).permute(0, 3, 1, 2) # 1, 6, 1024, 20

    elif delta and not neighbors:
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # 1, 1024, 20, 3
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2) # 1, 6, 1024, 20
     
    return feature, abs_x





class DGCNN_Transformer(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Transformer, self).__init__()

        self.args = args
        self.k = args.k
        self.deg = args.deg
        self.delta = args.delta
        self.neighbors = args.neighbors
        self.rpe = args.rpe
        self.ape = args.ape
        self.scale = args.scale

        print("self.deg : {}".format(self.deg))
        print("self.delta : {}".format(self.delta))
        print("self.neighbors : {}".format(self.neighbors))
        print("self.rpe : {}".format(self.rpe))
        print("self.ape : {}".format(self.ape))
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        if self.deg:
            self.deg_bn1 = nn.BatchNorm1d(1024)
            self.deg_bn2 = nn.BatchNorm1d(1024)
            self.deg_bn3 = nn.BatchNorm1d(1024)
            self.deg_bn4 = nn.BatchNorm1d(1024)
        else:
            self.deg_bn1 = None
            self.deg_bn2 = None
            self.deg_bn3 = None
            self.deg_bn4 = None 

        
        if self.ape:
            # self.pos_nn1 is skipped 
            self.pos_nn2 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.Tanh(),
                                     nn.Conv2d(32, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64)) # B, 3, N, 1 -> B, C, N, 1

            self.pos_nn3 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.Tanh(),
                                     nn.Conv2d(32, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64)) # B, 3, N, 1 -> B, C, N, 1

            self.pos_nn4 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.Tanh(),
                                     nn.Conv2d(32, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(128)) # B, 3, N, 1 -> B, C, N, 1
        else:
            self.pos_nn2 = None
            self.pos_nn3 = None
            self.pos_nn4 = None                          


        self.relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = AttentionConv(3*2, 64, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale)
        self.conv2 = AttentionConv(64*2, 64, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale)
        self.conv3 = AttentionConv(64*2, 128, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale)
        self.conv4 = AttentionConv(128*2, 256, kernel_size=self.k, groups=8, rpe=self.rpe, scale=self.scale)
        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2)) # 64 + 64 + 128 + 256 = 512

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        # shape of x : batch, feature, npoints, neighbors
        # convolution(shared mlp to xi, xj-xi & max)
        # =>
        # transformer(shared wq, wk, wv to xi)

        if self.ape:
            ptcld = x
        else:
            ptcld = None

        #print("1")
        x, abs_x = get_neighbors(x, k=self.k, deg=self.deg, delta=self.delta, neighbors=self.neighbors, bn=self.deg_bn1) # b, 3, 1024, 20
        x1 = self.conv1(x, abs_x) # b, 64, 1024
        x1 = self.relu(self.bn1(x1)).squeeze(3)

        #print("2")
        x, abs_x = get_neighbors(x1, k=self.k, deg=self.deg, delta=self.delta, neighbors=self.neighbors, bn=self.deg_bn2, ape=self.pos_nn2, ptcld=ptcld) # b, 64, 1024, 20
        x2 = self.conv2(x, abs_x) # b, 64, 1024
        x2 = self.relu(self.bn2(x2)).squeeze(3)

        #print("3")
        x, abs_x = get_neighbors(x2, k=self.k, deg=self.deg, delta=self.delta, neighbors=self.neighbors, bn=self.deg_bn3, ape=self.pos_nn3, ptcld=ptcld) # b, 128, 1024, 20
        x3 = self.conv3(x, abs_x) # b, 128, 1024
        x3 = self.relu(self.bn3(x3)).squeeze(3)

        #print("4")
        x, abs_x = get_neighbors(x3, k=self.k, deg=self.deg, delta=self.delta, neighbors=self.neighbors, bn=self.deg_bn4, ape=self.pos_nn4, ptcld=ptcld) # b, 256, 1024, 20
        x4 = self.conv4(x, abs_x) # b, 256, 1024, 20
        x4 = self.relu(self.bn4(x4)).squeeze(3)

        x = self.conv5(x4)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x



