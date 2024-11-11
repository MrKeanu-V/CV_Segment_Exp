from dataloaders.datasets import pascal
from dataloaders.datasets.crack500 import Crack500
from torch.utils.data import DataLoader, random_split


def make_dataloader(args, **kwargs):
    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'crack500':
        # divide the dataset into train/val/test sets according to the given ratio.
        full_set = Crack500(args, type='all')
        train_ratio, val_ratio, test_ratio = args.divide_ratio
        train_size = int(train_ratio * len(full_set))
        val_size = int(val_ratio * len(full_set))
        test_size = len(full_set) - train_size - val_size
        train_set, val_set, test_set = random_split(full_set, [train_size, val_size, test_size])  # use manual_seed to ensure reproducibility

        num_class = full_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
