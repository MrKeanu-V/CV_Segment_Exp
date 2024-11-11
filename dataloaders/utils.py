import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from mypath import Pather
import random
import glob
import shutil
from torchvision.utils import make_grid


def move_samples(samples, source_dir, target_dir):
    for sample in samples:
        image_path = os.path.join(source_dir, sample + '.jpg')
        mask_path = os.path.join(source_dir, sample + '.png')

        target_image_path = os.path.join(target_dir, sample + '.jpg')
        target_mask_path = os.path.join(target_dir, sample + '.png')

        shutil.copy(image_path, target_image_path)  # 不改变原格式
        shutil.copy(mask_path, target_mask_path)


def dataset_divide(platform: str, dataset: str, divide_rate):
    """ Divide dataset """
    # Set Divide Ratio
    assert abs(sum(divide_rate) - 1.0) < 1e-9, "数据集划分比例必须和为1"  # float有误差
    train_ratio = divide_rate[0]  # 训练集比例
    val_ratio = divide_rate[1]  # 验证集比例
    # test_ratio = divide_rate[2]  # dont need

    data_dir = Pather.db_root_dir(platform, dataset)
    cur_data_dir = os.path.join(os.getcwd(), 'dataset', dataset)
    if os.path.exists(cur_data_dir):
        shutil.rmtree(cur_data_dir)  # delete all things
    os.makedirs(cur_data_dir, exist_ok=False)
    train_dir = os.path.join(cur_data_dir, 'train')
    val_dir = os.path.join(cur_data_dir, 'val')
    test_dir = os.path.join(cur_data_dir, 'test')
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(val_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)

    # 从原始数据集中进行划分
    im_ids = []
    images = glob.glob(os.path.join(data_dir, '*.jpg'))
    for i, image in enumerate(images):
        line = os.path.splitext(os.path.basename(image))[0]
        im_ids.append(line)

    num_samples = len(im_ids)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    # 随机打乱
    random.shuffle(im_ids)
    # 依次划分
    train_set = im_ids[:num_train]
    val_set = im_ids[num_train:num_train + num_val]
    test_set = im_ids[num_train + num_val:]

    move_samples(train_set, data_dir, train_dir)
    move_samples(val_set, data_dir, val_dir)
    move_samples(test_set, data_dir, test_dir)

    # 误差检验
    train_len, val_len, test_len = len(train_set), len(val_set), len(test_set)
    print("Ori samples: {} tran_set samples: {} val_set samples: {} test_set samples: {} total samples {}".format(num_samples, train_len, val_len, test_len, train_len+val_len+test_len))


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """
    Decode segmentation class labels into a color image.
    Args:
        label_mask: (H,W) ndarray of integer values denoting the class label at each spatial location.
        dataset: str, dataset name.
        plot: bool, whether to show the resulting color image.

    Returns:
        rgb: (H,W,3) ndarray of color image.
    """
    n_classes, label_colours = get_labels(dataset)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
        return rgb
    else:
        return rgb


def encode_segmap(dataset, mask):
    """
    Encode segmentation label images as class numbers.
    Args:
        dataset: str, dataset name.
        mask: (C,H,W) ndarray of integer values denoting

    Returns:
        label_mask: (H,W) ndarray of integer values denoting the class label at each spatial location.
    """
    if dataset == 'crack500':
        threshold = 0.3
        np.where(mask > threshold, 1, 0)
    else:
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(get_labels(dataset).keys()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
    return label_mask


def get_labels(dataset: str):
    """
    Get color labels and number of classes for a dataset.
    Args:
        dataset: str, dataset name.

    Returns:
         n_classes: int, number of classes.
         label_colours: (n_classes, 3) ndarray with RGB values for each class.
    """
    if dataset == 'crack500':
        n_classes = 2
        return n_classes, np.asarray([[0, 0, 0], [255, 0, 0]])  # Red means crack, Black is background
    elif dataset == 'pascal':
        n_classes = 21
        return n_classes, np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])
    else:
        raise NotImplementedError('Dataset {} not available.'.format(dataset))


def print_sample(sample, tensor=False):
    image, label = sample['image'], sample['label']
    if tensor:
        image = sample['image'].ToPILImage()
        label = sample['label'].ToPILImage()

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, gridspec_kw={'hspace': 0.4})

    axes[0].imshow(image)
    axes[0].set_title('Image')

    axes[1].imshow(label, cmap=plt.cm.binary_r)  # 二值映射
    axes[1].set_title('Label')

    plt.show()
