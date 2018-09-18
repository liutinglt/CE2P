import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.6, min_kept=100000):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        # self.min_kept_ratio = float(min_kept_ratio)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        min_kept = self.min_kept #int(self.min_kept_ratio * n * h * w)
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class DiscriminativeLoss(nn.Module):

    def __init__(self, thea, delta, ignore_label=255):
        super(DiscriminativeLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thea = thea
        self.delta = delta
        self.relu = nn.ReLU(inplace=True)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        ntarget = target.data.cpu().numpy()
        predict = predict.permute(0,2,3,1)
        cls_ids = np.unique(ntarget)
        # cls_ids = cls_ids[cls_ids != 0]
        cls_ids = cls_ids[cls_ids != self.ignore_label]
        cls_ids = [cls_id for cls_id in cls_ids if np.sum(ntarget == cls_id) > 20]
        centers = {}
        loss_var = 0
        for cls_id in cls_ids:
            index = (target == cls_id)
            index = index.unsqueeze(3)
            cls_prediction = predict[index].view((-1,c))
            mean = cls_prediction.mean(0)
            centers[cls_id] = mean
            result = self.relu(torch.norm(mean - cls_prediction, 2, 1) - self.thea)
            # result = torch.max(result, Variable(torch.FloatTensor([self.thea]).cuda(), requires_grad=False))
            loss_var += torch.pow(result, 2).mean()
            # print('cls_id: {}, loss_dis: {}'.format(cls_id, result.data.cpu().numpy().shape))
        loss_var /= len(cls_ids)

        loss_dis = 0
        for f_cls_id in cls_ids:
            for s_cls_id in cls_ids:
                if f_cls_id != s_cls_id:
                    result = self.relu(2*self.delta - torch.norm(centers[f_cls_id]-centers[s_cls_id]))
                    loss_dis += torch.pow(result, 2)
        loss_dis /= max((len(cls_ids) * (len(cls_ids) - 1)), 1)

        loss_reg = 0
        for cls_id in cls_ids:
            loss_reg += torch.norm(centers[cls_id])
        loss_reg /= len(cls_ids)

        # print('loss_var: {},loss_dis: {}, loss_reg:{}'.format(loss_var.data.cpu().numpy(), loss_dis.data.cpu().numpy(), 0.001*loss_reg.data.cpu().numpy()))
        return  loss_var + loss_dis + 0.001 * loss_reg


class HNMDiscriminativeLoss(nn.Module):

    def __init__(self, thea, delta, ignore_label=255):
        super(HNMDiscriminativeLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thea = thea
        self.delta = delta
        self.relu = nn.ReLU(inplace=True)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        ntarget = target.data.cpu().numpy()
        predict = predict.permute(0,2,3,1)
        cls_ids = np.unique(ntarget)
        # cls_ids = cls_ids[cls_ids != 0]
        cls_ids = cls_ids[cls_ids != self.ignore_label]
        cls_ids = [cls_id for cls_id in cls_ids if np.sum(ntarget == cls_id) > 20]
        centers = {}
        loss_var = 0
        for cls_id in cls_ids:
            index = (target == cls_id)
            index = index.unsqueeze(3)
            cls_prediction = predict[index].view((-1,c))
            mean = cls_prediction.mean(0)
            centers[cls_id] = mean
            result = self.relu(torch.norm(mean - cls_prediction, 2, 1) - self.thea)
            # result = torch.max(result, Variable(torch.FloatTensor([self.thea]).cuda(), requires_grad=False))
            normliaze = max(np.sum(result.data.cpu().numpy() > 0), 1)
            # print(normliaze, result.size())
            loss_var += torch.pow(result, 2).sum() / normliaze
            # print('cls_id: {}, loss_dis: {}'.format(cls_id, result.data.cpu().numpy().shape))
        loss_var /= len(cls_ids)

        loss_dis = 0
        for f_cls_id in cls_ids:
            for s_cls_id in cls_ids:
                if f_cls_id != s_cls_id:
                    result = self.relu(2*self.delta - torch.norm(centers[f_cls_id]-centers[s_cls_id]))
                    loss_dis += torch.pow(result, 2)
        loss_dis /= (len(cls_ids) * (len(cls_ids) - 1))

        loss_reg = 0
        for cls_id in cls_ids:
            loss_reg += torch.norm(centers[cls_id])
        loss_reg /= len(cls_ids)

        # print('loss_var: {},loss_dis: {}, loss_reg:{}'.format(loss_var.data.cpu().numpy(), loss_dis.data.cpu().numpy(), 0.001*loss_reg.data.cpu().numpy()))
        return  loss_var + loss_dis + 0.001 * loss_reg
