import torch.nn as nn
import torch


class SegmentationLosses(object):
    def __init__(self, weight=None, batch_average=False, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):  # 构造损失函数
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bce':
            return self.BCEWithLogitsLoss
        else:
            raise NotImplementedError

    def BCEWithLogitsLoss(self, logit, target):  # 结合Sigmoid层的BCELoss
        n, c, h, w = logit.size()
        criterion = nn.BCEWithLogitsLoss(weight=self.weight)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLoss(self, logit, target):  # 交叉熵损失函数
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):  # 焦点损失函数，用于解决类别数量不平衡的损失函数
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


if __name__ == '__main__':
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 2, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    c = torch.BoolTensor(1, 7, 7).cuda()  # 布尔张量运算时会被转换为整数类型
    t1 = loss.CrossEntropyLoss(a, b).item()
    t2 = loss.CrossEntropyLoss(a, c).item()
    print(loss.CrossEntropyLoss(a, b).item())
    print(t2)
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
