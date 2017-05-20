import torch.utils.data as data
from skimage import color

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".npy"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class FolderWithImages(data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None, load_img_fn=load_img):
        super(FolderWithImages, self).__init__()

        self.image_filenames = [join(root, x)
                                for x in listdir(root) if is_image_file(x.lower())]

        self.load_img = load_img_fn
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = self.load_img(self.image_filenames[index])

        if self.input_transform:
            input = self.input_transform(input)
        # if self.target_transform:
        #     target = self.target_transform(target)

        # to rgb
        
        t = input.numpy().transpose(1, 2, 0).astype(np.float64)
        
        color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])
        target = torch.FloatTensor(color.lab2rgb(t).transpose(2, 0, 1))

        # target = input.clone()  # Image.open(filepath).convert('RGB')
        return input, target

    def __len__(self):
        return len(self.image_filenames)
