import torch.nn as nn
# import encoding.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .loss import DiscriminativeLoss, HNMDiscriminativeLoss, OhemCrossEntropy2d
 


class Criterion(nn.Module):
    def __init__(self, ignore_index=255):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = OhemCrossEntropy2d(ignore_index, 0.7, 100000)
        self.criterion2 = HNMDiscriminativeLoss(0.5, 1.5, ignore_index)

    def forward(self, preds, target):
        assert len(preds) == 2
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)


        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion2(scale_pred, target)

        print(loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0])
        return loss1 + loss2
     
    
class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target)

        return loss
    
class CriterionCrossEntropyEdge(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropyEdge, self).__init__()
        self.ignore_index = ignore_index
          
    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        
        input_labels = target.data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels==1).astype(np.float)
        neg_num = np.sum(input_labels==0).astype(np.float)
        
        weight_pos = neg_num/(pos_num+neg_num)
        weight_neg = pos_num/(pos_num+neg_num)
        weights = (weight_neg, weight_pos)  
        weights = Variable(torch.from_numpy(np.array(weights)).float().cuda())
        
        scale_pred1 = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = F.cross_entropy(scale_pred1, target, weights )
        scale_pred2 = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = F.cross_entropy(scale_pred2, target, weights )
        scale_pred3 = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss3 = F.cross_entropy(scale_pred3, target, weights )
        scale_pred4 = F.upsample(input=preds[3], size=(h, w), mode='bilinear')
        loss4 = F.cross_entropy(scale_pred4, target, weights )
        scale_pred5 = F.upsample(input=preds[4], size=(h, w), mode='bilinear')
        loss5 = F.cross_entropy(scale_pred5, target, weights ) 
        scale_pred6 = F.upsample(input=preds[5], size=(h, w), mode='bilinear')
        loss6 = F.cross_entropy(scale_pred6, target, weights ) 

        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 
    
class CriterionCrossEntropyEdgeParsing(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropyEdgeParsing, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
          
    def forward(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        
        input_labels = target[1].data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels==1).astype(np.float)
        neg_num = np.sum(input_labels==0).astype(np.float)
        
        weight_pos = neg_num/(pos_num+neg_num)
        weight_neg = pos_num/(pos_num+neg_num)
        weights = (weight_neg, weight_pos)  
        weights = Variable(torch.from_numpy(np.array(weights)).float().cuda())
        
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target[0])
        
        scale_pred1 = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred1, target[0])
     
        scale_pred2 = F.upsample(input=preds[2], size=(h, w), mode='bilinear')
        loss2 = F.cross_entropy(scale_pred2, target[1], weights )
#         scale_pred3 = F.upsample(input=preds[3], size=(h, w), mode='bilinear')
#         loss3 = F.cross_entropy(scale_pred3, target[1], weights )
#         scale_pred4 = F.upsample(input=preds[4], size=(h, w), mode='bilinear')
#         loss4 = F.cross_entropy(scale_pred4, target[1], weights )
#         scale_pred5 = F.upsample(input=preds[5], size=(h, w), mode='bilinear')
#         loss5 = F.cross_entropy(scale_pred5, target[1], weights ) 

#         return loss+loss1+loss2 +loss3 +loss4 +loss5 
        return loss+loss1+loss2
    
class CriterionMSE(nn.Module):
    def __init__(self):
        super(CriterionMSE, self).__init__() 
        self.criterion = torch.nn.MSELoss()
        
    def forward(self, preds,  target): 
        h, w = preds.size(2), preds.size(3) 
        scale_target = F.upsample(input=target, size=(h, w), mode='bilinear')
        loss = self.criterion(preds, scale_target)
        #loss = self.criterion(preds, target)
        return loss
        
class CriterionCrossEntropyPose(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropyPose, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.criterion2 = torch.nn.MSELoss()  
         

    def forward(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)
        parsing_out1 = preds[0]
        parsing_out2 = preds[1]
        parsing_out3 = preds[2]
        pose_out1 = preds[3]
        pose_out2 = preds[4]
        pose_out3 = preds[5]
  
        scale_pred1 = F.upsample(input=parsing_out1, size=(h, w), mode='bilinear')
        loss_parsing1 = self.criterion(scale_pred1, target[0])
        scale_pred2 = F.upsample(input=parsing_out2, size=(h, w), mode='bilinear')
        loss_parsing2 = self.criterion(scale_pred2, target[0])
        scale_pred3 = F.upsample(input=parsing_out3, size=(h, w), mode='bilinear')
        loss_parsing3 = self.criterion(scale_pred3, target[0])
        
        
        scale_pose_out1 = F.upsample(input=pose_out1, size=(h, w), mode='bilinear')
        loss_pose1 = self.criterion2(scale_pose_out1, target[1])
        scale_pose_out2 = F.upsample(input=pose_out2, size=(h, w), mode='bilinear')
        loss_pose2 = self.criterion2(scale_pose_out2, target[1])
        scale_pose_out3 = F.upsample(input=pose_out3, size=(h, w), mode='bilinear')
        loss_pose3 = self.criterion2(scale_pose_out3, target[1])
        
          
        return loss_parsing1 + loss_parsing2 + loss_parsing3+loss_pose1+loss_pose2+loss_pose3
    
class CriterionOhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, thres=0.6, min_kept=200000):
        super(CriterionOhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        # 1/10 of the pixels within a mini-batch, if we use 2x4 on two cards, it should be 200000
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept)

    def forward(self, preds, target):
        # assert len(preds) == 2
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target)
        # print('OhemCrossEntropy2d Loss: {}'.format(loss.data.cpu().numpy()[0]))
        return loss
