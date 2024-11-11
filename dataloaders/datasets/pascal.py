import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Pather
from torchvision import transforms
from dataloaders import custom_transforms as transformer

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self, args, base_dir=None, split='train'):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super.__init__()
        if base_dir is None:
            base_dir = Pather.get_db_dir(args.platform, 'pascal')
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for split in self.split:
            with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for i, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + '.png')
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_train(sample)
            elif split == "val":
                return self.transform_val(sample)

    def transform_train(self,sample):
        composed_transforms = transforms.Compose([
            transformer.RandomRotate(),
            transformer.Normalize(mean=(0.485, 0.456, 0.406),td=(0.229, 0.224, 0.225)),
            transformer.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self,sample):
        composed_transforms = transforms.Compose([
            transformer.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transformer.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split={})'.format(self.split)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')
    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        for j in range(sample['image'.size()[0]]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[j]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[j], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if i == 1:
            break

        plt.show(block=True)