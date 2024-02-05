import os
import numpy as np

import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import glob
import random
from torch.utils.data import Dataset
from PIL import Image


########################################################
# Methods for Image DataLoader
########################################################


def convert_to_rgb(image):
    """
    Convert an image to RGB format.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: RGB converted image.

    """
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        # print("self.files_B ", self.files_B)
        """ Will print below array with all file names
        ['/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-06-26 14:04:52.jpg',
        '/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-08-02 09:19:52.jpg',..]
        """

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        # a % b => a is divided by b, and the remainder of that division is returned.

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = convert_to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = convert_to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        # Finally ruturn a dict
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


########################################################
# Replay Buffer
########################################################




class ReplayBuffer:
   

    # We keep an image buffer that stores
    # the 50 previously created images.
    def __init__(self, max_size=50):
       
        assert max_size > 0, "Empty buffer."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                # If the buffer is full, decide whether to replace an old image in the buffer with the new one,
                # and whether to add the new image or an old image to the return batch.
                if random.uniform(0, 1) > 0.5:
                    # With a 50% chance, replace an old image in the buffer with the new image
                    # and add the old image to the return batch.
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[
                        i
                    ] = element  # replaces the older image with the newly generated image.
                else:
                    # With a 50% chance, keep the buffer as is and add the new image to the return batch.
                    to_return.append(element)
        return Variable(torch.cat(to_return))


########################################################
# Learning Rate scheduling with `lr_lambda`
########################################################


class LambdaLR:
   
    def __init__(self, n_epochs, offset, decay_start_epoch):
       
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        
        # Below line checks whether the current epoch has exceeded the decay epoch(which is 100)
        # e.g. if current epoch is 80 then max (0, 80 - 100) will be 0.
        # i.e. then entire numerator will be 0 - so 1 - 0 is 1
        # i.e. the original LR remains as it is.
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


########################################################
# Initialize convolution layer weights to N(0,0.02)
########################################################


def initialize_conv_weights_normal(m):
   
    # Extract the class name of the module to determine its type.
    classname = m.__class__.__name__

    # Check if the module is a Convolutional layer.
    if classname.find("Conv") != -1:
        # Initialize the weights of the Convolutional layer using a normal distribution
        # with mean 0.0 and standard deviation 0.02.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        # Check if the Convolutional layer has a bias attribute and is not None.
        if hasattr(m, "bias") and m.bias is not None:
            # Initialize the biases of the Convolutional layer as constant 0.
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Check if the module is a Batch Normalization layer.
    elif classname.find("BatchNorm2d") != -1:
        # Initialize the weights of the Batch Normalization layer using a normal distribution
        # with mean 1.0 and standard deviation 0.02.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        # Initialize the biases of the Batch Normalization layer as constant 0.
        torch.nn.init.constant_(m.bias.data, 0.0)
