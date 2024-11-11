import os
import shutil
import torch
import glob
from collections import OrderedDict


class Saver(object):
    """
    Saver class to save and load checkpoints.
    Args:
        args: arguments from command line.
    Attributes:
        directory: directory to save checkpoints.
        experiment_dir: directory to save current experiment.
        runs: list of previous experiments. Used to update Save Directory.
    Note:
        The directory is structured as follows:
        runs/dataset/checkname/experiment_*/checkpoints/checkpoint.pth.tar
        run_id: id of current experiment.
        best_pred: best mIoU.
        previous_miou: list of previous mIoU.
        max_miou: maximum mIoU.
    """

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('runs', args.dataset, args.checkname)  # 生成保存路径
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0  # 根据最后目录的id生成

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best: bool, filename: str = 'checkpoint.pth.tar'):
        """
        Save checkpoint to disk.

        Args:
            state: Model state to save.
            is_best: Is current checkpoint the best one.
            filename: filename of checkpoint.pth.tar. Default: 'checkpoint.pth.tar'
        """
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:  # 更新mIoU
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        """
        Save experiment configuration to disk.
        """
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()  # 有序字典为了后续输出
        p['datset'] = self.args.dataset
        if hasattr(self.args, 'backbone'):
            p['backbone'] = self.args.backbone  # only in model with backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
