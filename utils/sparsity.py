# -*- coding: utf-8 -*-
"""sparsity.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M0F2FNCOFYC-ZHj_jaYZlErAPqB3QwGe
"""

import torch
import torch.nn as nn

def clip_lower_weights(layer, threshold):
    with torch.no_grad():
        weights = layer.weight.data.view(-1)
        mask = (weights < threshold) & (-threshold < weights)
        weights[mask] = 0.0
        layer.weight.data = weights.view(layer.weight.data.shape)

def layer_clips(model, thresh):
    for layer in model.features.children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            clip_lower_weights(layer, thresh)
    for layer in model.classifier.children():
        if isinstance(layer, nn.Linear):
            clip_lower_weights(layer, thresh)

def check_sparseness(model):
    total_params = 0
    sparse_params = 0
    trail = []
    sparse_list = []

    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:
            total_params += torch.numel(param)
            sparse_params += torch.sum(torch.abs(param) == 0).item()
            sparse_list.append(torch.sum(torch.abs(param) == 0).item()*100 / torch.numel(param))

    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:
            trail_sparse = 0
            if len(param.size()) == 4:
                for i in range(param.size()[0]):
                    if torch.sum(torch.abs(param[-i-1])) == 0:
                        trail_sparse += 1
                    else:
                        break
            else:
                for i in range(param.size()[0]):
                    if torch.sum(torch.abs(param[-i-1])) == 0:
                        trail_sparse += 1
                    else:
                        break
            trail.append(trail_sparse)
    sparsity = sparse_params / total_params
    return sparsity, trail, sparse_list
