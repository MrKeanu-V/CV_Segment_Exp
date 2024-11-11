import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory  # directory=Saver.directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):  # 需要添加Tag用标识所属大类
        """ Use SSH Tunnel to Open Tensorboard at local Browser """
        grid_image = make_grid(image.clone().cpu().data, nrow=4, padding=2, normalize=False)
        writer.add_image('Image', grid_image, global_step=global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), nrow=4, normalize=False, value_range=(0, 255))
        # 原版：grid_image = make_grid(decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(),
        #                                               dataset=dataset), nrow=4, global_step=global_step, normalize=False, value_range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(),
                                                       dataset=dataset), nrow=4, padding=2, normalize=False, value_range=(0, 255))
        writer.add_image('GroundTruth label', grid_image, global_step)

    def visualize_images(self, writer, tag, images, global_step):
        """ Log images on tensorboard"""
        writer.add_images(tag, images, global_step, dataformats='NCHW')  # 传参规范，不能前加关键字，中间不加，要么全加，要么其后加

    # 未使用
    def visualize_labels(self, writer, tag, labels, global_step):
        writer.add_images(tag, labels, global_step, dataformats='NHW')
        size = labels.shape  # 获得的labels形状为NHW，已经失去了通道信息

        t = labels[:3]  # 去掉一层了
        temp_1 = torch.squeeze(t, 1)  # 无变化
        temp = torch.squeeze(labels[:3], 1).detach().cpu().numpy()
        writer.add_image('Test/temp', temp, global_step)
        grid_image = make_grid(decode_seg_map_sequence(temp, dataset='Crack500'), nrow=4, padding=2, normalize=False,
                               range=(0, 255))
        # grid_image = make_grid(labels.unsqueeze(1), nrow=4, padding=2, normalize=False)
        writer.add_image(tag, grid_image, global_step)

    def log_hparams(self, acc, mIoU):  # 未检验
        hparam_dict = {}
        metric_dict = {'metrics/accuracy': acc, 'metrics': mIoU}
        self.writer.add_hparams(hparam_dict, metric_dict)


if __name__=='__main__':
    from train import  get_args
    from models.DSCNet import DSCNet
    from dataloaders.datasets.crack500 import Crack500
    from torch.utils.data import DataLoader
    from utils.summaries import TensorboardSummary
    import numpy as np

    args = get_args()
    smy_dir = os.path.join(os.getcwd(), 'runs', 'test')
    summary = TensorboardSummary(directory=smy_dir)
    writer = summary.create_summary()
    # test_dataset = Crack500(args, split='test')
    # test_dataloader = DataLoader(test_dataset, 4, shuffle=None)

    t_model = DSCNet(n_channels=3, n_classes=2)

    # 加载预训练权重
    runs_dir = os.path.join(os.path.dirname(os.getcwd()), 'runs', args.dataset, 'DSCNet')  # , 'experiment_1'
    tar_dir = os.path.join(runs_dir, 'model_best.pth.tar')  # 'model_best.pth.tar' 'checkpoint.pth.tar'
    checkpoint = torch.load(tar_dir)    # tar文件不用解压，可以直接使用torch.load加载为map映射
    t_model.load_state_dict(checkpoint, strict=False)
    t_model.eval()

    # Z = np.random.rand(4, 5, 6, 7)
    # T = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
    # print(Z)
    # A = Z.astype(dtype=np.float32)
    # A = torch.from_numpy(Z)
    # print(A.shape)
    # O = torch.max(A, 1).numpy()
    # B = torch.max(A, 1)[0].detach().cpu().numpy()
    # C = torch.max(A, 1)[1].detach().cpu().numpy()
    # torch.max(A, 1)[1].detach().cpu().numpy()

    a = torch.randn(4,2,3,3)
    aa = a.numpy()
    print(a)
    b = torch.max(a)
    bb = b.numpy()
    print(b)
    c = torch.max(a, 1)
    cc = c.numpy()
    print(c)

    writer.close()
