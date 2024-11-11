import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']		# 很有启发，以此获取了Image 和 Label并对其进行预处理

        if random.random() >= 0.5:
            image = image[::-1]			# ::-1用于翻转序列  有点问题
            label = label[::-1]

        h, w = image.shape[:2]		# :2前两项

        if isinstance(self.output_size, int):  # 不一样的变换方式
            if h > w:  # 等比例缩放
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w))
        lbl = label.resize((new_h, new_w))

        return {'image': img, 'label': lbl}


class Normalize(object):
    """
    Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # 转化为numpy类型后图像色调就发生了变化，因为Image中颜色为RGB，而在Numpy中的顺序为BGR。不是数据类型的原因
        img = np.array(img).astype(np.float32)  # Current dont change datatype .astype(np.float32)必须是float32，否则转换为torch会被识别为double类型，导致与卷积bias类型不同而报错  10月7日更新实际应为float64否则颜色会出错
        mask = np.array(mask)  # 标签数据通常为整数值

        # Normalize
        img /= 255.0
        # img -= self.mean  # 引起颜色变化的主要原因
        # img /= self.std  # 目的是是数据符合正态分布，更利于模型训练

        return {'image': img, 'label': mask}


class ToTensor(object):
    """Convert ndarray in sample to Tensors. """

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).transpose((2, 0, 1))  # swap color axis because the difference in Np and Tensor
        mask = np.array(mask)

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = img.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask}


class RandomRotate(object):
    def __init__(self,degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image':img, 'mask': mask}


class RandomCrop(object):  # 没问题了
    def __init__(self, base_size, crop_size):
        self.base_size = base_size
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

    def __call__(self, sample):
        img, mask = sample["image"], sample['label']

        if random.random() >= 0.5:  # 随机水平翻转
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        w, h = img.size  # This is PIL.Image.size first return width
        new_h, new_w = self.crop_size

        top = np.random.randint(0, max(1, h - new_h))  # 避免h和new_h相等时报错
        left = np.random.randint(0, max(1, w - new_w))

        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return {'image': img, 'label': mask}


class RandomScaleCrop(object):  # 裁剪算法有问题
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

