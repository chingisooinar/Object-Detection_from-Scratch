#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:29:25 2021

@author: nuvilabs
"""
import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.l_relu = nn.LeakyReLU(0.1)
        self.use_bn = bn_act
    def forward(self, x):
        if self.use_bn:
            return self.l_relu(self.bn(self.conv(x)))
        else:
            return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(repeats):
            self.layers += [
                    nn.Sequential(
                        CNNBlock(channels, channels//2, kernel_size=1),
                        CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                    )
                ]
        self.use_residual = use_residual
        self.repeats = repeats
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x
                
        

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, classes):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
                CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
                CNNBlock(2*in_channels, (classes + 5) * 3, bn_act=False, kernel_size=1)
            )
        self.classes = classes
    def forward(self, x):
        return (self.pred(x)
                .reshape(x.shape[0], 3, self.classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2))
    
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    
    def forward(self, x):
        outputs = []
        route_connections = []
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels, 
                                       out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels
            
            elif isinstance(module, list):
                layers.append(ResidualBlock(in_channels, repeats=module[1]))
            
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels *= 3
        return layers
       

