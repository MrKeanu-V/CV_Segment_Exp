import os
import copy
import cv2
import colorsys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from models.u2net import U2Net
from models.UNet import UNet
from dataloaders.utils import decode_segmap


class Predicter:
    def __init__(self, args):
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.classify_type = args.classify_type
        self.mode = args.mode
        self.model = self.load_model()
        self.device = args.device
        self.crop = args.crop
        self.output_dir = os.path.join(os.getcwd(), 'outputs', self.model_name)

    def load_model(self):
        if self.model_name == "unet":
            model = UNet(n_channels=3, n_classes=2)
        elif self.model_name == "u2net":
            model = U2Net(n_channels=3, n_classes=2)
        else:
            raise NotImplementedError("Model: {} is not implemented.".format(self.model_name))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(self.model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        return model

    def detect_image(self, image_path, crop=False):
        image = Image.open(image_path).convert('RGB')
        if crop:
            image = self.crop_image(image)
        image = self.transform_image(image)
        logit = self.model(image)
        rgb = decode_segmap(logit, self.classify_type, plot=True)
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(os.path.join(self.output_dir, os.path.basename(image)))
        return rgb

    def transform_image(self, image):
        np_image = np.array(image)
        np_image = np_image / 255.0
        np_image = np.transpose(np_image, (2, 0, 1))
        torch_image = torch.from_numpy(np_image).float().unsqueeze(0)
        torch_image = torch_image.to(self.device)
        return torch_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='u2net', choices=['u2net', 'unet'], help='which model to use')
    parser.add_argument('--classify_type', type=str, default='crack500', help='path of the model')
    parser.add_argument('--mode', type=str, default='dir_predict', choices=['predict', 'dir_predict'],
                        help='which mode to predict')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='which device to use')
    parser.add_argument('--crop', type=bool, default=False, help='crop or not')
    parser.add_argument('--count', type=bool, default=False, help='count or not')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.model_path = os.path.join(os.getcwd(), 'runs/Crack500/u2net/model_best.pth.tar')
    predicter = Predicter(args)
    # the path of the image to predict
    detect_img_path = "img/test.jpg"
    predicter.detect_image(detect_img_path, crop=args.crop)

