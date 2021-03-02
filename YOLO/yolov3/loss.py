#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:55:49 2021

@author: nuvilabs
"""

import random
import torch
import torch.nn as nn

from utils import intersection_over_union

class YOLOv3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        #Constants
        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1
    def forward(self, preds, target, anchors):
        obj = target[..., 0] == 1 # in paper this is Iobj_i
        noobj = target[..., 0] == 0 # in paper this is Inoobj_i
        #No object loss
        no_object_loss = self.bce(
            (preds[..., 0:1][noobj]), (target[..., 0:1][noobj])   
        )
        #object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) #p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(preds[..., 1:3]), anchors * torch.exp(preds[..., 3:5])], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        
        object_loss = self.bce((preds[..., 0:1][obj]),(ious * target[..., 0:1][obj]))        
        #box coordinate loss
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        target[..., 3:5] = torch.log( 1e-16 + target[..., 3:5] / anchors)#inverse
        box_loss = self.mse((preds[..., 1:5][obj]), (target[..., 1:5][obj]))
        
        #class loss
        class_loss = self.entropy((preds[...,5:][obj]), (target[..., 5][obj].long()))
        
        return self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss