# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import mermaid.finite_differences as fdt
from torch.autograd import Variable
from math import exp

###############################################################################
# Functions
###############################################################################

class NCCLoss(nn.Module):
    """
    A implementation of the normalized cross correlation (NCC)
    """
    def forward(self,input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1).view(input.shape[0],1) + 1e-10
        target_minus_mean = target - torch.mean(target, 1).view(input.shape[0],1) + 1e-10
        nccSqr = ((input_minus_mean * target_minus_mean).mean(1)) / torch.sqrt(
                    ((input_minus_mean ** 2).mean(1)) * ((target_minus_mean ** 2).mean(1)))
        nccSqr =  nccSqr.mean()

        assert not torch.isnan(nccSqr), 'NCC loss is Nan.'

        return (1 - nccSqr)

class NGFLoss(torch.nn.Module):
    def __init__(self):
        super(NGFLoss,self).__init__()
        self.eps = 1e-10
    
    def forward(self, I0, I1):
        g_I0 = self._image_normalized_gradient(I0)
        g_I1 = self._image_normalized_gradient(I1)
        
        sim_per_pix = 1. - torch.mean(torch.bmm(g_I0.view(-1,1,2), g_I1.view(-1,2,1))**2)
        return sim_per_pix

    def _image_normalized_gradient(self, x):
        '''
        :param x. BxCxWxH
        '''
        g_x = F.pad(x[:,:,2:,:] - x[:,:,0:-2,:], (0,0,1,1), "constant", 0)
        g_y = F.pad(x[:,:,:,2:] - x[:,:,:,0:-2], (1,1,0,0), "constant", 0)

        # Use linear conditioin
        g_x[:,:,0:1,:] =  (x[:,:,1:2,:] - x[:,:,0:1,:])
        g_x[:,:,-1:,:] =  (x[:,:,-1:,:] - x[:,:,-2:-1,:])
        g_y[:,:,:,0:1] =  (x[:,:,:,1:2] - x[:,:,:,0:1])
        g_y[:,:,:,-1:] =  (x[:,:,:,-1:] - x[:,:,:,-2:-1])

        g = torch.stack([g_x, g_y], dim=-1)
        g = g/torch.sqrt(torch.sum(g**2, dim=-1, keepdim=True)+self.eps)
        return g