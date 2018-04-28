#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:34:27 2018

@author: bai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def Ternarize(tensor):
    output = torch.zeros(tensor.size())
    delta = Delta(tensor)
    alpha = Alpha(tensor,delta)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1,-1):
            pos_one = (w > delta[i]).type(torch.cuda.FloatTensor)
            neg_one = -1 * (w < -delta[i]).type(torch.cuda.FloatTensor)
            out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
        output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
    return output
            

def Alpha(tensor,delta):
        Alpha = []
        for i in range(tensor.size()[0]):
            count = 0
            abssum = 0
            absvalue = tensor[i].view(1,-1).abs()
            for w in absvalue:
                truth_value = w > delta[i] #print to see
            count = truth_value.sum()
            abssum = torch.matmul(absvalue,truth_value.type(torch.cuda.FloatTensor).view(-1,1))
            Alpha.append(abssum/count)
        alpha = Alpha[0]
        for i in range(len(Alpha) - 1):
            alpha = torch.cat((alpha,Alpha[i+1]))
        return alpha

def Delta(tensor):
    n = tensor[0].nelement()
    if(len(tensor.size()) == 4):     #convolution layer
        delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
    elif(len(tensor.size()) == 2):   #fc layer
        delta = 0.7 * tensor.norm(1,1).div(n)
    return delta
            

class TernaryLinear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(TernaryLinear,self).__init__(*args,**kwargs)
    def forward(self,input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.linear(input,self.weight,self.bias)
        return out

class TernaryConv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(TernaryConv2d,self).__init__(*args,**kwargs)
    def forward(self,input):
        self.weight.data = Ternarize(self.weight.data)
        out = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return out

class LeNet5_T(nn.Module):
    def __init__(self):
        super(LeNet5_T,self).__init__()
        self.conv1 = TernaryConv2d(1,32,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(32,64,kernel_size = 5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = TernaryLinear(1024,512)
        self.fc2 = TernaryLinear(512,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x),2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x),2))
        x = x.view(-1,1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  
        
        
