import torch
import torchvision.transforms as T
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch import nn



def hello_helper():
    print("Hello from ps4_helper.py!")


SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

### Helper Functions for style transfer
"""
Our pretrained model was trained on images that had been preprocessed by subtracting
the per-color mean and dividing by the per-color standard deviation. We define a few
helper functions for performing and undoing this preprocessing.
"""


def preprocess(img, size=224):
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ]
    )
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled



def deprocess(img, should_rescale=True):
    # should_rescale true for style transfer
    transform = T.Compose(
        [
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
            T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
            T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
            T.ToPILImage(),
        ]
    )
    return transform(img)



# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy

    vnum = int(scipy.__version__.split(".")[1])
    major_vnum = int(scipy.__version__.split(".")[0])

    assert (
        vnum >= 16 or major_vnum >= 1
    ), "You must install SciPy >= 0.16.0 to complete this notebook."



def get_zero_one_masks(img, size):
    """
    Helper function to get [0, 1] mask from a mask PIL image (black and white).

    Inputs
    - img: a PIL image of shape (3, H, W)
    - size: image size after reshaping

    Returns: A torch tensor with values of 0 and 1 of shape (1, H, W)
    """
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
        ]
    )
    img = transform(img).sum(dim=0, keepdim=True)
    mask = torch.where(img < 1, 0, 1).to(torch.float)
    return mask
    
def show_images(images):
    images = torch.reshape(
        images, [images.shape[0], -1]
    )  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return

def count_params(model):
    """Count the number of parameters in the model"""
    param_count = sum([p.numel() for p in model.parameters()])
    return param_count
    
def initialize_weights(m):
    """Initializes the weights of a torch.nn model using xavier initialization"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)