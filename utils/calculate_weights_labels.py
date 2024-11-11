import os
import numpy as np
from tqdm import tqdm
from mypath import Pather


def calculate_weights_labels(dataset, dataloader, num_classes):
    """
    Calculate class weights for the dataset. The weights are Balanced Focal Loss weights.
    Args:
        dataset: which dataset to calculate weights for
        dataloader: pytorch dataloader for the dataset
        num_classes: number of classes in the dataset

    Returns: numpy array of class_weights
    """
    # Create Instance from dataloader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader, desc='Calculating classes weights')
    # print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    class_weights_path = os.path.join(Pather.db_root_dir(dataset), dataset + '_classes_weights.npy')
    np.save(class_weights_path, ret)

    return ret
