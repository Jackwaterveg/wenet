#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Label smoothing module."""

import torch
from torch import nn

import os
import numpy as np
root_dir = "../DeepSpeech/compare/result_store/wenet"

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """
    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

        print ("===========")
        print ("padding_idx", self.padding_idx)
        print ("self.confidence", self.confidence)
        print ("self.smoothing",self.smoothing)
        print ("self.size",self.size)
        print ("self.normalize_length", self.normalize_length)
        """
        padding_idx -1
        self.confidence 0.9
        self.smoothing 0.1
        self.size 4233
        self.normalize_length False
        """


    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        print ("label smoothing:target", target)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        print ("true_dist", true_dist)
        np.save(os.path.join(root_dir, "true_dist.npy"), true_dist.cpu().detach().numpy())
        
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        print ("x", x)
        print ("log_softmax", torch.log_softmax(x, dim=1))
        log_softmax = torch.log_softmax(x, dim=1)
        x_log_softmax_np = log_softmax.cpu().detach().numpy()
        np.save(os.path.join(root_dir, "log_softmax_.npy"), x_log_softmax_np)
        np.save("wenet_smoothing_x.npy", x.cpu().detach().numpy())
        np.save("wenet_log_softmax.npy", torch.log_softmax(x, dim=1).cpu().detach().numpy())
        denom = total if self.normalize_length else batch_size
        print ("ignore",ignore)
        res = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        res_np = res.cpu().detach().numpy()
        np.save(os.path.join(root_dir, "attn_res_.npy"), res_np)
        print ("attn_res_np", res_np)
        return res