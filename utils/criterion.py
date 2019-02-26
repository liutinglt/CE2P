import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d


class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
   
    def parsing_loss(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # loss for parsing
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])

        # loss for edge
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)

        return loss

    def forward(self, preds, target):
          
        loss = self.parsing_loss(preds, target) 
        return loss
    