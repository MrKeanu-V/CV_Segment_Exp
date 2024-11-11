import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.recall = 0.0
        self.precision = 0.0
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # 初始化混淆矩阵

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def _generate_matrix(self, gt_image, pre_image):  # 生成混淆矩阵
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def Pixel_Accuracy(self):  # 像素准确率=预测真像素点/总像素数目
        PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return PA

    # 精准率Precision本意是针对某一类别的，也作Class Precision=TP/(TP+FP)
    def Pixel_Accuracy_Class(self):  # 类别像素准确率（即Precision）CPA= TP / (TP + FP)
        CPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        CPA = np.nanmean(CPA)  # 忽略输入数组中的NaN值，并计算剩余元素的平均值
        self.precision = CPA
        return CPA

    def Mean_Pixel_Accuracy(self):  # 类别平均像素准确率MPA,当只有真假两类是MPA、CPA和Precision值相同。但实际项目里两者等价
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        MPA = np.nanmean(MPA)
        self.precision = MPA
        return MPA

    # 召回率用于评估正例样本中的覆盖能力，工业价值大
    def Recall(self):  # 召回率（仅限二分类问题）Recall=TP/(TP+FN)
        recall = np.diag(self.confusion_matrix)/self.confusion_matrix.sum(axis=0)
        recall = np.nanmean(recall)
        self.recall = recall
        return recall

    # F1分数用于评估精确性和召回性能之间的平衡情况
    def F1_score(self):  # F1分数（同仅限二分类问题）F1 = 2 * (Precision * Recall) / (Precision + Recall)即2*TP/(TP+FP+TP+FN)
        # f1 = 2 * (self.precision * self.recall)/(self.precision + self.recall + 1e-9)  # 此方法F1偏大
        f1 = 2*np.diag(self.confusion_matrix)/(self.confusion_matrix.sum(axis=0)+self.confusion_matrix.sum(axis=1))
        f1 = np.nanmean(f1)
        return f1

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def rand_confusion_matrix(self):
        self.confusion_matrix = np.random.randint(0, 20, (2, 2))
