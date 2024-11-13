import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from mypath import Pather
from utils.saver import Saver
from models.u2net import U2Net
from models.UNet import UNet
from dataloaders import make_dataloader
from utils.summaries import TensorboardSummary
from utils.calculate_weights_labels import calculate_weights_labels
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from dataloaders.utils import dataset_divide
from models.sync_batchnorm.replicate import patch_replication_callback


class Trainer(object):
    def __init__(self, args):
        # args.cuda = False
        self.args = args

        # Define Saver from Utils
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary from Utils
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_dataloader(args, **kwargs)

        # Define network
        if self.args.model == 'unet':
            model = UNet(n_channels=3, n_classes=self.nclass)
        elif self.args.model == 'u2net':
            model = U2Net(n_channels=3, n_classes=self.nclass)
        else:
            raise NotImplementedError

        # Define Optimizer
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=args.nesterov)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),
                                          weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weight_path = os.path.join(Pather.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weight_path):
                weight = np.load(classes_weight_path)
            else:
                weight = calculate_weights_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight)  # weight.astype(np.float32)
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Define Learning Rate Scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Set Device
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming Checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])  # for DataParallel
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader, desc=f'Epoch: {epoch + 1}/{self.args.epochs} training',
                    total=len(self.train_loader) * self.args.batch_size, unit='images', colour='green', ncols=150)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label'],
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output = self.model(image)
            # Calculate the loss between pred and gt
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            tbar.set_description('Epoch: {}/{} training'.format(epoch + 1, self.args.epochs))
            tbar.set_postfix_str('loss: %.3f' % (train_loss / (i + 1)))
            tbar.update(self.args.batch_size)

            # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #     global_step = i + num_img_tr * epoch
            #     self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        # Refresh the tbar
        tbar.n = tbar.total
        tbar.refresh()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f'Epoch: {epoch + 1}/{self.args.epochs} validating',
                    total=len(self.val_loader) * self.args.batch_size, unit='images', colour='green', ncols=150)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label'],
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

            tbar.set_description('Epoch: {}/{} validating'.format(epoch + 1, self.args.epochs))
            tbar.set_postfix_str('loss: %.3f' % (test_loss / (i + 1)))
            tbar.update(self.args.batch_size)
        # Refresh the tbar
        tbar.n = tbar.total
        tbar.refresh()
        # Fast test during the training
        mean_pixel_Acc = self.evaluator.Mean_Pixel_Accuracy()
        recall = self.evaluator.Recall()
        f1_score = self.evaluator.F1_score()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/mean_pixel_Acc', mean_pixel_Acc, epoch)
        self.writer.add_scalar('val/Recall', recall, epoch)
        self.writer.add_scalar('val/F1_Score', f1_score, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/FWIoU', FWIoU, epoch)
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (
            epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Mean_Classes_Pixel_Accuracy:{}, Recall:{}, F1:{}, mIoU:{}, fwIoU: {}".format(mean_pixel_Acc, recall,
                                                                                            f1_score, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def get_args():
    parser = argparse.ArgumentParser(description="Pytorch U2Net Training")
    parser.add_argument('--model', type=str, default='u2net', choices=['u2net', 'unet'])
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='crack500',
                        choices=['crack500', 'pascal'], help='which dataset to use (default: Crack500)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')  # Only use when dataset is coco
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')  # if OOM warning, set workers=0
    parser.add_argument('--base_size', type=int, default=320,
                        help='base image size')  # 这里有问题，图片大小非矩形
    parser.add_argument('--crop-size', type=int, default=360,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')

    # optimizer params
    parser.add_argument('--optimizer', type=str, default=None, choices=['Adam', 'SGD'], help='which optimizer to use (default: Adam)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        metavar='M', help='w-decay (default: optimizer`s default value)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # dataset divide
    parser.add_argument('--platform', default='windows', help='which platform to run code',
                        choices=['windows', 'remote', 'linux'])
    parser.add_argument('--divide-ratio', nargs=2, type=float, default=[0.7, 0.2],
                        help='how to divide dataset in train, val, test')

    return parser.parse_args()


def main():
    args = get_args()
    # lower case
    args.dataset = args.dataset.lower()
    args.model = args.model.lower()
    args.platform = args.platform.lower()

    # set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    print('Using GPU: {}  Number of GPU: {}'.format(args.cuda, args.gpu_ids))

    # default setting for epochs, batch_size and lr
    if args.epochs is None:
        epochs = {
            'crack500': 50,
            'pascal': 50,
        }
        args.epochs = epochs[args.dataset]  # epochs[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    # set optimizer type
    if args.optimizer is None:
        args.optimizer = 'Adam'
    args.optimizer = args.optimizer.lower()

    # default setting for lr
    if args.lr is None:
        if args.optimizer == 'adam':
            lrs = {
                'crack500': 0.001,
                'pascal': 0.0007,
            }
        elif args.optimizer == 'SGD':
            lrs = {
                'crack500': 0.01,
                'pascal': 0.007,
            }
        args.lr = lrs[args.dataset] / (4 * len(args.gpu_ids)) * args.batch_size

    # set weight_decay
    if args.weight_decay is None:
        if args.optimizer == 'adam':
            args.weight_decay = 1e-2
        elif args.optimizer == 'SGD':
            args.weight_decay = 1e-4

    # set Checkpoint Name
    if args.checkname is None:
        args.checkname = args.model

    # dataset divide
    args.divide_ratio = [args.divide_ratio[0], args.divide_ratio[1], 1 - sum(args.divide_ratio)]
    print(f"divide_ratio: {args.divide_ratio}")
    assert args.divide_ratio[2] >= 0, 'the sum of divide_rate should be equal to 1'
    args.is_dir_divide = False  # 是否需要对数据集进行文件夹级别的划分，默认不划分
    if args.is_dir_divide:
        dataset_divide(args.platform, args.dataset, args.divide_rate)

    # print args
    index = 1
    for k, v in args.__dict__.items():
        print(f"{index} \t {k}: {v}")
        index += 1

    # set random seed and train
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoch:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == '__main__':
    main()
