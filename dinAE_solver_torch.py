#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:46:18 2020

@author: rfablet
"""

import torch
#import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constrained Conv2D Layer with zero-weight at central point
class ConstrainedConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ConstrainedConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        with torch.no_grad():
          self.weight[:,:,int(self.weight.size(2)/2)+1,int(self.weight.size(3)/2)+1] = 0.0
    def forward(self, input):
        return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

## Check the constraint applies
#if 1*1:
#  conv1 = ConstrainedConv2d(1,10,(3,3),padding=0)
#
#  print(conv1.weight[:,0,2,2])
#  conv1.weight[:,0,2,2] = 0.
#  inputs = torch.randn(1,1,28,28)
#  y = conv1(torch.autograd.Variable(inputs))
#  print(conv1.weight[:,0,2,2])
  
## ResNet architecture (Conv2d)      
class ResNetConv2D(torch.nn.Module):
  def __init__(self,Nblocks,dim,K,
                 kernel_size,
                 padding=0):
      super(ResNetConv2D, self).__init__()
      self.resnet = self._make_ResNet(Nblocks,dim,K,kernel_size,padding)

  def _make_ResNet(self,Nblocks,dim,K,kernel_size,padding):
      layers = []
      for kk in range(0,Nblocks):
        layers.append(torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False))
        layers.append(torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False))

      return torch.nn.Sequential(*layers)


  def forward(self, x):
      x = self.resnet ( x )

      return x
        
# Pytorch ConvLSTM2D
class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class ConvLSTM1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv1d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

#if torch.cuda.is_available():
#  torch.set_default_tensor_type('torch.cuda.FloatTensor')

## fixed-point solver for the end-to-end learning of AE model with missing data
class Model_AE_FP(torch.nn.Module):
    def __init__(self,mod_AE,NiterProjection):
        super(Model_AE_FP, self).__init__()
        self.model_AE = mod_AE
        self.NProj    = NiterProjection
    def forward(self, x_inp,mask):
        #mask   = torch.add(1.0,torch.mul(mask,0.0)) # set mask to 1 # debug
        mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        x      = torch.mul(x_inp,1.0)
        for kk in range(0,self.NProj):
          x_proj = self.model_AE(x)
          x_proj = torch.mul(x_proj,mask_)
          x      = torch.mul(x, mask)   
          x      = torch.add(x , x_proj )
        x_proj        = self.model_AE(x)
        return x_proj

## fixed-point + gradient-based solver for the end-to-end learning of AE model with missing data
# Bug: does not work if no FP iteration is applied (NiterProjection=0)
class Model_AE_GradFP(torch.nn.Module):
    def __init__(self,mod_AE,ShapeData,NiterProjection,NiterGrad,GradType,OptimType):
    #def __init__(self,mod_AE,GradType,OptimType):
        super(Model_AE_GradFP, self).__init__()
        self.model_AE = mod_AE

        with torch.no_grad():
            self.GradType = GradType
            self.OptimType = OptimType
            self.NProjFP   = int(NiterProjection)
            self.NGrad     = int(NiterGrad)
            self.shape     = ShapeData
        
        if len(self.shape) == 2: ## 1D Data
            if self.OptimType == 0: # fixed-step gradient descent
              self.conv1    = torch.nn.Conv1d(self.shape[0], self.shape[0],1, padding=0)              
            elif self.OptimType == 1: # ConvNet gradient descent using previous and current gradient
              self.conv1    = torch.nn.Conv1d(2*self.shape[0], 8*self.shape[0],3, padding=1)
              self.conv2    = torch.nn.Conv1d(8*self.shape[0], 16*self.shape[0],3, padding=1)
              self.conv3    = torch.nn.Conv1d(16*self.shape[0], self.shape[0],3, padding=1)
            elif self.OptimType == 2: # ConvNet gradient descent using previous and current gradient
              self.lstm1    = ConvLSTM1d(self.shape[0],5*self.shape[0],3)
              self.conv1    = torch.nn.Conv1d(5*self.shape[0], self.shape[0], 1, padding=0)           
        elif len(self.shape) == 3: ## 2D Data            
            if self.OptimType == 0: # fixed-step gradient descent
              self.conv1    = torch.nn.Conv2d(self.shape[0], self.shape[0], (1,1), padding=0)
            elif self.OptimType == 1: # ConvNet gradient descent using previous and current gradient
              self.conv1    = torch.nn.Conv2d(2*self.shape[0], 8*self.shape[0], (3,3), padding=1)
              self.conv2    = torch.nn.Conv2d(8*self.shape[0], 16*self.shape[0], (3,3), padding=1)
              self.conv3    = torch.nn.Conv2d(16*self.shape[0], self.shape[0], (3,3), padding=1)
            elif self.OptimType == 2: # ConvNet gradient descent using previous and current gradient
              self.lstm1    = ConvLSTM2d(self.shape[0],5*self.shape[0],3)
              self.conv1    = torch.nn.Conv2d(5*self.shape[0], self.shape[0], (1,1), padding=0)
    def forward(self, x_inp,mask):
        #mask   = torch.add(1.0,torch.mul(mask,0.0)) # set mask to 1 # debug
        mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        x      = torch.mul(x_inp,1.0)

        # fixed-point iterations
        if self.NProjFP > 0:
          for kk in range(0,self.NProjFP):
        #if NiterProjection > 0:
        #  x      = torch.mul(x_inp,1.0)
        #  for kk in range(0,NiterProjection):            
            x_proj = self.model_AE(x)
            x_proj = torch.mul(x_proj,mask_)
            x      = torch.mul(x, mask)   
            x      = torch.add(x , x_proj )

        # gradient iteration
        if self.NGrad > 0:
          for kk in range(0,self.NGrad):
        #if NiterGrad > 0:
        #  for kk in range(0,NiterGrad):
            # compute gradient
            if self.GradType == 0: ## subgradient
              grad = torch.add(self.model_AE(x),-1.,x)
            else: ## true gradient using autograd
              loss = torch.sum( torch.add(self.model_AE(x),-1.,x)**2 )
              grad = torch.autograd.grad(loss,x)[0]
            #grad = grad.view(-1,1,self.shape[1],self.shape[2])
            grad.retain_grad()

            # gradient step

            if self.OptimType == 0: # fixed-step gradient
              grad = self.conv1( grad )
            elif self.OptimType == 1: # convNet for grad using previous and current gradient
              if kk == 0:
                grad_old = torch.randn(grad.size()).to(device)
              gradAll  = torch.cat((grad_old,grad),1)
              grad_old = torch.mul(1.,grad)

              grad = self.conv1( gradAll )
              grad = self.conv2( F.relu( grad ) )
              grad = self.conv3( F.relu( grad ) )
              #grad = grad.view(-1,self.shape[1],self.shape[2])

            elif self.OptimType == 2: # convLSTLM suing grad as input
              if kk == 0:
                hidden,cell = self.lstm1(grad,None)
              else:
                hidden,cell = self.lstm1(grad,[hidden,cell])
              grad = self.conv1(hidden)

            # update
            #grad  = grad.view(-1,self.shape[1],self.shape[2])
            x_new = torch.add(x,grad)
            x_new = torch.mul(x_new,mask_)
            x     = torch.mul(x, mask)   
            x     = torch.add(x , x_new )

        x_proj        = self.model_AE(x)
        return x_proj
