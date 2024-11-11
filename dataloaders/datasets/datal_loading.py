import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename).convert('RGB')


class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __init__(self, num_classes: int, img_dir: str, mask_dir: str, img_ext: str = '.jpg', mask_ext: str = '.png'):
        super().__init__()
        self.num_classes = num_classes
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else Path(img_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        self.ids = [splitext(file)[0] for file in listdir(img_dir) if
                    isfile(join(img_dir, file)) and file.endswith(img_ext)]
        if not self.ids:
            raise RuntimeError(f"No images found in {img_dir}, make sure you put your images there")

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.images = [join(self.img_dir, f'{img_id}{img_ext}') for img_id in self.ids]
        self.masks = [join(self.mask_dir, f'{img_id}{mask_ext}') for img_id in self.ids]

        assert len(self.images) == len(self.masks), "Number of images and masks should be equal"
        # display stats
        logging.info(f'Number of images in {self.img_dir}: {len(self.ids)}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        _img = load_image(img_path)
        _mask = load_image(mask_path)

        if self.transform is not None:
            _img = self.transform(_img)
            _mask = self.transform(_mask)
        else:
            _img = self.default_transform()(_img)
            _mask = self.default_transform()(_mask)

        return {'image': _img, 'mask': _mask}

    @staticmethod
    def default_transform():
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        return transform
