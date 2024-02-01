import numpy as np
import itertools
import time
import datetime
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch

from matplotlib.pyplot import figure
from IPython.display import clear_output

from PIL import Image
import matplotlib.image as mpimg

from utils import *
from cyclegan import *

cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


##############################################
# Defining all hyperparameters
##############################################


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=19,
    lambda_cyc=10.0,
    lambda_id=5.0,
)

##############################################
# Setting Root Path for Google Drive or Kaggle
##############################################

# Root Path for Google Drive
root_path = "/content/drive/MyDrive/All_Datasets/ct_mri"

# Root Path for Kaggle
# root_path = '../input/summer2winter-yosemite'


########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(size, size))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()





def to_img(x):
  
    # Reshape the input tensor to have the dimensions:
    # (batch_size * 2, num_channels, height, width)
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


def plot_output(path, x, y):
   
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()


##############################################
# Defining Image Transforms to apply
##############################################
transforms_ = [
    transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_=transforms_),
    batch_size=hp.batch_size,
    shuffle=True,
    num_workers=1,
)
val_dataloader = DataLoader(
    ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_=transforms_),
    batch_size=16,
    shuffle=True,
    num_workers=1,
)

##############################################
# SAMPLING IMAGES
##############################################


def save_img_samples(batches_done):
    """Saves a generated sample from the test set"""
    print("batches_done ", batches_done)
    imgs = next(iter(val_dataloader))

    Gen_CT_MRI.eval()
    Gen_MRI_CT.eval()

    real_CT = Variable(imgs["A"].type(Tensor))
    fake_MRI = Gen_CT_MRI(real_CT)
    real_MRI = Variable(imgs["B"].type(Tensor))
    fake_CT = Gen_MRI_CT(real_MRI)
    # Arange images along x-axis
    real_CT = make_grid(real_CT, nrow=16, normalize=True)
    real_MRI = make_grid(real_MRI, nrow=16, normalize=True)
    fake_CT = make_grid(fake_CT, nrow=16, normalize=True)
    fake_MRI = make_grid(fake_MRI, nrow=16, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_CT, fake_MRI, real_MRI, fake_CT), 1)

    path = root_path + "/%s.png" % (batches_done)  # Path when running in Google Colab

    # path =  '/kaggle/working' + "/%s.png" % (batches_done)    # Path when running inside Kaggle
    save_image(image_grid, path, normalize=False)
    return path


##############################################
# SETUP, LOSS, INITIALIZE MODELS and BUFFERS
##############################################

# Creating criterion object (Loss Function) that will
# measure the error between the prediction and the target.
criterion_GAN = torch.nn.MSELoss()

criterion_cycle = torch.nn.L1Loss()

criterion_identity = torch.nn.L1Loss()

input_shape = (hp.channels, hp.img_size, hp.img_size)

##############################################
# Initialize generator and discriminator
##############################################

Gen_CT_MRI = GeneratorResNet(input_shape, hp.num_residual_blocks)
Gen_MRI_CT = GeneratorResNet(input_shape, hp.num_residual_blocks)

Disc_CT = Discriminator(input_shape)
Disc_MRI = Discriminator(input_shape)

if cuda:
    Gen_CT_MRI = Gen_CT_MRI.cuda()
    Gen_MRI_CT = Gen_MRI_CT.cuda()
    Disc_CT = Disc_CT.cuda()
    Disc_MRI = Disc_MRI.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

##############################################
# Initialize weights
##############################################

Gen_CT_MRI.apply(initialize_conv_weights_normal)
Gen_MRI_CT.apply(initialize_conv_weights_normal)

Disc_CT.apply(initialize_conv_weights_normal)
Disc_MRI.apply(initialize_conv_weights_normal)


##############################################
# Buffers of previously generated samples
##############################################

fake_CT_buffer = ReplayBuffer()

fake_MRI_buffer = ReplayBuffer()


##############################################
# Defining all Optimizers
##############################################
optimizer_G = torch.optim.Adam(
    itertools.chain(Gen_CT_MRI.parameters(), Gen_MRI_CT.parameters()),
    lr=hp.lr,
    betas=(hp.b1, hp.b2),
)
optimizer_Disc_CT = torch.optim.Adam(Disc_CT.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))

optimizer_Disc_MRI = torch.optim.Adam(Disc_MRI.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))


##############################################
# Learning rate update schedulers
##############################################
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step
)

lr_scheduler_Disc_CT = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_CT,
    lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)

lr_scheduler_Disc_MRI = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_MRI,
    lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)
############################################
# Save Checkpoint Function
############################################
def save_checkpoint(epoch, batches_done):
    #checkpoint_path = f"../checkpoint_epoch{epoch}_batch{batches_done}.pth"
    checkpoint_path = f"/content/drive/MyDrive/All_Datasets/ct_mri/checkpoint_epoch{epoch}_batch{batches_done}.pth"

    torch.save(
        {
            "epoch": epoch,
            "batches_done": batches_done,
            "Gen_MRI_CT_state_dict": Gen_MRI_CT.state_dict(),
            "Gen_CT_MRI_state_dict": Gen_CT_MRI.state_dict(),
            "Disc_CT_state_dict": Disc_CT.state_dict(),
            "Disc_MRI_state_dict": Disc_MRI.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_Disc_CT_state_dict": optimizer_Disc_CT.state_dict(),
            "optimizer_Disc_MRI_state_dict": optimizer_Disc_MRI.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved at {checkpoint_path}")

#############################################
# Load CheckPoint
#############################################
def load_checkpoint(checkpoint_path, Gen_MRI_CT, Gen_CT_MRI, Disc_CT, Disc_MRI, optimizer_G, optimizer_Disc_CT, optimizer_Disc_MRI):
    checkpoint = torch.load(checkpoint_path)

    Gen_MRI_CT.load_state_dict(checkpoint["Gen_MRI_CT_state_dict"])
    Gen_CT_MRI.load_state_dict(checkpoint["Gen_CT_MRI_state_dict"])
    Disc_CT.load_state_dict(checkpoint["Disc_CT_state_dict"])
    Disc_MRI.load_state_dict(checkpoint["Disc_MRI_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_Disc_CT.load_state_dict(checkpoint["optimizer_Disc_CT_state_dict"])
    optimizer_Disc_MRI.load_state_dict(checkpoint["optimizer_Disc_MRI_state_dict"])

    epoch = checkpoint["epoch"]
    batches_done = checkpoint["batches_done"]

    return epoch, batches_done
##############################################
# Final Training Function
##############################################


def train(
    Gen_MRI_CT,
    Gen_CT_MRI,
    Disc_CT,
    Disc_MRI,
    train_dataloader,
    n_epochs,
    criterion_identity,
    criterion_cycle,
    lambda_cyc,
    criterion_GAN,
    optimizer_G,
    fake_CT_buffer,
    fake_MRI_buffer,
    clear_output,
    optimizer_Disc_CT,
    optimizer_Disc_MRI,
    Tensor,
    sample_interval,
    lambda_id,
):
    # TRAINING
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):

            # Set model input
            real_CT = Variable(batch["A"].type(Tensor))
            real_MRI = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths i.e. target vectors
            # 1 for real images and 0 for fake generated images
            valid = Variable(
                Tensor(np.ones((real_CT.size(0), *Disc_CT.output_shape))),
                requires_grad=False,
            )

            fake = Variable(
                Tensor(np.zeros((real_CT.size(0), *Disc_CT.output_shape))),
                requires_grad=False,
            )

            #########################
            #  Train Generators
            #########################

            Gen_CT_MRI.train()
            Gen_MRI_CT.train()

            optimizer_G.zero_grad()

            
            loss_id_CT = criterion_identity(Gen_MRI_CT(real_CT), real_CT)

           
            loss_id_MRI = criterion_identity(Gen_CT_MRI(real_MRI), real_MRI)

            loss_identity = (loss_id_CT + loss_id_MRI) / 2

            # GAN losses for GAN_CT_MRI
            fake_MRI = Gen_CT_MRI(real_CT)

            loss_GAN_CT_MRI = criterion_GAN(Disc_MRI(fake_MRI), valid)

            # GAN losses for GAN_MRI_CT
            fake_CT = Gen_MRI_CT(real_MRI)

            loss_GAN_MRI_CT = criterion_GAN(Disc_CT(fake_CT), valid)

            loss_GAN = (loss_GAN_CT_MRI + loss_GAN_MRI_CT) / 2

            # Cycle Consistency losses
            reconstructed_CT = Gen_MRI_CT(fake_MRI)
         
            loss_cycle_CT = criterion_cycle(reconstructed_CT, real_CT)

            reconstructed_MRI = Gen_CT_MRI(fake_CT)

            loss_cycle_MRI = criterion_cycle(reconstructed_MRI, real_MRI)

            loss_cycle = (loss_cycle_CT + loss_cycle_MRI) / 2

            
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()

            
            optimizer_G.step()

            #########################
            #  Train Discriminator CT
            #########################

            optimizer_Disc_CT.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_CT(real_CT), valid)
            # Fake loss (on batch of previously generated samples)

            fake_CT_ = fake_CT_buffer.push_and_pop(fake_CT)

            loss_fake = criterion_GAN(Disc_CT(fake_CT_.detach()), fake)

            
            loss_Disc_CT = (loss_real + loss_fake) / 2

            
            loss_Disc_CT.backward()

            
            optimizer_Disc_CT.step()

            #########################
            #  Train Discriminator MRI
            #########################

            optimizer_Disc_MRI.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_MRI(real_MRI), valid)

            # Fake loss (on batch of previously generated samples)
            fake_MRI_ = fake_MRI_buffer.push_and_pop(fake_MRI)

            loss_fake = criterion_GAN(Disc_MRI(fake_MRI_.detach()), fake)

            
            loss_Disc_MRI = (loss_real + loss_fake) / 2

           
            loss_Disc_MRI.backward()

            
            optimizer_Disc_MRI.step()

            loss_D = (loss_Disc_CT + loss_Disc_MRI) / 2

            

            batches_done = epoch * len(train_dataloader) + i

            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                clear_output()
                plot_output(save_img_samples(batches_done), 30, 40)
            if(batches_done%100==0):
                save_checkpoint(epoch, batches_done)

#############################################
#Load Checkpoints
#############################################
    # Specify the checkpoint path
checkpoint_path = "/content/drive/MyDrive/All_Datasets/ct_mri/checkpoint_epoch0_batch100.pth"

# Initialize or load your models and optimizers
# (assuming you have already defined Gen_MRI_CT, Gen_CT_MRI, Disc_CT, Disc_MRI, optimizer_G, optimizer_Disc_CT, optimizer_Disc_MRI)
    #epoch, batches_done = 0, 0  # Set initial values

# Check if a checkpoint file exists
    #if os.path.exists(checkpoint_path):
    # Load the checkpoint
      #hp.epoch, batches_done = load_checkpoint(checkpoint_path, Gen_MRI_CT, Gen_CT_MRI, Disc_CT, Disc_MRI, optimizer_G, optimizer_Disc_CT, optimizer_Disc_MRI)

models_initialized = False

# If models haven't been initialized, load the checkpoints
if not models_initialized and os.path.exists(checkpoint_path):
    # Load the checkpoint
    hp.epoch,batches_done = load_checkpoint(checkpoint_path, Gen_MRI_CT, Gen_CT_MRI, Disc_CT, Disc_MRI, optimizer_G, optimizer_Disc_CT, optimizer_Disc_MRI)
    print("Check point Loaded epocha and batch:",hp.epoch,batches_done)
    # Set the flag to indicate that models have been initialized
    models_initialized = True
# Continue with training
##############################################
# Execute the Final Training Function
##############################################

train(
    Gen_MRI_CT=Gen_MRI_CT,
    Gen_CT_MRI=Gen_CT_MRI,
    Disc_CT=Disc_CT,
    Disc_MRI=Disc_MRI,
    train_dataloader=train_dataloader,
    n_epochs=hp.n_epochs,
    criterion_identity=criterion_identity,
    criterion_cycle=criterion_cycle,
    lambda_cyc=hp.lambda_cyc,
    criterion_GAN=criterion_GAN,
    optimizer_G=optimizer_G,
    fake_CT_buffer=fake_CT_buffer,
    fake_MRI_buffer=fake_MRI_buffer,
    clear_output=clear_output,
    optimizer_Disc_CT=optimizer_Disc_CT,
    optimizer_Disc_MRI=optimizer_Disc_MRI,
    Tensor=Tensor,
    sample_interval=hp.sample_interval,
    lambda_id=hp.lambda_id,
)
