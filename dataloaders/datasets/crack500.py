import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
# import matplotlib.pyplot as plt

from mypath import Pather


class Crack500(Dataset):
    """
    Crack500 dataset. This dataset is used for binary segmentation.

    Args:
        args: parameters
        split: split of dataset. Default: 'train'

    Attributes:
        _base_dir: base directory of dataset
        split: split of dataset
        im_ids: image ids
        images: images paths
        masks: masks paths , it's the ground truth of segmentation
        threshold: threshold for binarization of mask
        NUM_CLASSES: number of classes
    """
    NUM_CLASSES = 2

    def __init__(self, args, type='all', transform=None):
        super().__init__()
        self.args = args
        self.type = type
        self._base_dir = Pather.db_root_dir(args.platform, args.dataset)

        self.im_ids = []
        self.images = []
        self.masks = []
        self.threshold = 0.3
        self.transform = transform
        self.default_transform = transforms.Compose([
            transforms.Resize((360, 360)),
            transforms.ToTensor(),
        ])

        images = glob.glob(os.path.join(self._base_dir, '*' + '.jpg'))
        for i, image in enumerate(images):
            line = image.split('.')[0]
            target = os.path.join(line + '.png')
            assert os.path.isfile(image)
            assert os.path.isfile(target)
            self.im_ids.append(line)
            self.images.append(image)
            self.masks.append(target)

        assert (len(self.images) == len(self.masks))

        # Display stats
        print('Number of images in Crack500 {}: {:d}'.format(type, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        _img = Image.open(self.images[idx]).convert('RGB')
        _temp = Image.open(self.masks[idx]).convert('L')
        # encode segmentation mask to 0 and 1
        _temp = self.encode_segmap(np.array(_temp))
        _mask = Image.fromarray(_temp)

        if self.transform is not None:
            _img = self.transform(_img)
            _mask = self.transform(_mask)
        else:
            _img = self.default_transform(_img)
            _mask = self.default_transform(_mask)

        return {'image': _img, 'label': _mask.squeeze()}

    def __str__(self):
        return 'Crack500 Dataset (cur split={})'.format(self.type)

    def encode_segmap(self, mask):
        return np.where(mask > self.threshold, 1, 0)

    # def decode_segmap(self, label):


if __name__ == '__main__':
    # Test the datasetloader
    # import sys
    # sys.path.insert(0, 'D:\Code\CV_Segment_Exp')

    from train import get_args
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import threading
    import concurrent.futures


    def display_sample(sample: dict):
        _image, _label = sample['image'], sample['label']
        to_pil = transforms.ToPILImage()
        images = [to_pil(img) for img in _image.cpu()]
        labels = [to_pil(lbl.cpu()) for lbl in _label]
        fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
        for img, lbl, ax in zip(images, labels, axs):
            ax.imshow(img)
            ax.imshow(lbl, alpha=0.5)
            # ax.axis('off')
        plt.show()


    from dataloaders.utils import decode_segmap

    args = get_args()
    dataset = Crack500(args=args)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, sample in enumerate(tqdm(dataloader, desc='Training Testing')):
            if i % 100 == 0:
                # executor.submit(display_sample, sample)
                _image, _label = sample['image'], sample['label']
                to_pil = transforms.ToPILImage()
                images = [to_pil(img) for img in _image.cpu()]
                labels = [to_pil(decode_segmap(lbl.cpu().numpy(), 'crack500')) for lbl in _label]
                fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
                for img, lbl, ax in zip(images, labels, axs):
                    ax.imshow(img)
                    ax.imshow(lbl, alpha=0.5)
                    # ax.axis('off')
                plt.show()
