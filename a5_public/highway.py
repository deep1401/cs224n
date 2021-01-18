#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """ Highway network for ConvNN
           - Relu
           - Sigmoid
           - gating mechanism from LSTM
       """

    def __init__(self, emb_size):
        super(Highway, self).__init__()
        self.projection = nn.Linear(emb_size, emb_size)
        self.gate = nn.Linear(emb_size, emb_size)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
                   Take mini-batch of sentence of ConvNN
                   @param x_conv_out:torch.Tensor : Tensor with shape (max_sentence_length, batch_size, embed_size)
                   @return x_highway:torch.Tensor : combined output with shape (max_sentence_length, batch_size, embed_size)
               """

        x_projection = F.relu(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = torch.mul(x_projection, x_gate) + torch.mul((1 - x_gate), x_conv_out)
        return x_highway

### END YOUR CODE
