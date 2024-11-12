import os
import copy
import cv2
import colorsys
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
from models.u2net import U2Net
from models.UNet import UNet
from models.sync_batchnorm.replicate import patch_replication_callback
from dataloaders.utils import decode_segmap


class Predicter:
    def __init__(self, args):
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.dataset = args.dataset
        self.device = args.device
        self.mode = args.mode
        self.output_dir = args.output_dir

        self.transform_image = transforms.Compose([
            transforms.Resize((360, 360)),
            transforms.ToTensor(),
        ])

        # set device
        if self.device == 'cuda':
            if torch.cuda.device_count() > 1:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = [torch.cuda.current_device()]
        self.model = self.load_model()

        # delete the output directory if it exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            del_list = os.listdir(self.output_dir)
            for f in del_list:
                os.remove(os.path.join(self.output_dir, f))

    def load_model(self):
        if self.dataset == "crack500":
            num_classes = 2
            num_channels = 3
        else:
            raise NotImplementedError("Dataset: {} is not implemented.".format(self.dataset))

        if self.model_name == "unet":
            model = UNet(n_channels=num_channels, n_classes=num_classes)
        elif self.model_name == "u2net":
            model = U2Net(n_channels=num_channels, n_classes=num_classes)
        else:
            raise NotImplementedError("Model: {} is not implemented.".format(self.model_name))

        checkpoint = torch.load(self.model_path)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)
            patch_replication_callback(model)
            model = model.cuda()
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()

        return model

    def predict(self, img_path, transform=None):
        if os.path.isfile(img_path):
            self.detect_image(img_path, transform)
        elif os.path.isdir(img_path):
            image_list = os.listdir(img_path)
            for img_name in image_list:
                img = os.path.join(img_path, img_name)
                self.detect_image(img, transform)
        else:
            raise FileNotFoundError("Image file:{} or directory not found.".format(img_path))

    def detect_image(self, img_path, transform=None):
        image = Image.open(img_path).convert('RGB')
        if transform is not None:
            img_tensor = transform(image)
        else:
            img_tensor = self.transform_image(image)

        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            output = self.model(img_tensor)
            output = output.squeeze(0)
            _, output = torch.max(output, 0)
            output_np = output.cpu().numpy()
            rgb = decode_segmap(output_np, self.dataset)
            rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb_img = rgb_img.resize(image.size, Image.BICUBIC)
            rgb_img.save(os.path.join(self.output_dir,  os.path.basename(img_path)))
        return rgb

    @staticmethod
    def resize_image(image, distort=False):
        image = np.array(image)
        size = [640, 360]
        iw, ih = image.shape[0:2][::-1]
        w, h = size
        if distort:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)

        return new_image




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='u2net', choices=['u2net', 'unet'], help='which model to use')
    parser.add_argument('--model-path', type=str, default='checkpoints', choices=['checkpoints', 'runs'], help='which type of model to use')
    parser.add_argument('--dataset', type=str, default='crack500', help='which dataset the model is trained on')
    parser.add_argument('--mode', type=str, default='dir_predict', choices=['predict', 'dir_predict'],
                        help='which mode to predict')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='which device to use')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.model_path == 'checkpoints':
        file_name = args.model_name.lower() + '_' + args.dataset.lower() + '_best.pth.tar'
        args.model_path = os.path.join(os.getcwd(), 'checkpoints', file_name)
    else:
        args.model_path = os.path.join(os.getcwd(), 'runs', args.dataset.lower(), args.model_name.lower(), 'model_best.pth.tar')
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError("Model Checkpoint file not found.")

    args.output_dir = os.path.join(os.getcwd(), 'results/outputs', args.model_name.lower())

    for k, v in vars(args).items():
        print(k, ':', v)

    predicter = Predicter(args)

    # the path of the image to predict
    if args.mode == 'predict':
        image_path = "results/images/test.jpg"
    elif args.mode == 'dir_predict':
        image_path = "results/images"
    else:
        raise NotImplementedError("Mode: {} is not implemented.".format(args.mode))

    predicter.predict(image_path)

