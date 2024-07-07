import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

"""
General utilities to help with implementation
"""


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return





def detection_visualizer(img, idx_to_class, bbox=None, pred=None, points=None):
    """
    Data visualizer on the original image. Support both GT
    box input and proposal input.

    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates
            (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional),
            a tensor of shape N'x6, where N' is the number
            of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    # Convert image to HWC if it is passed as a Tensor (0-1, CHW).
    if isinstance(img, torch.Tensor):
        img = (img * 255).permute(1, 2, 0)

    img_copy = np.array(img).astype("uint8")
    _, ax = plt.subplots(frameon=False)

    ax.axis("off")
    ax.imshow(img_copy)

    # fmt: off
    if points is not None:
        points_x = [t[0] for t in points]
        points_y = [t[1] for t in points]
        ax.scatter(points_x, points_y, color="yellow", s=24)

    if bbox is not None:
        for single_bbox in bbox:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(1.0, 0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                ax.text(
                    x0, y0, obj_cls, size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )

    if pred is not None:
        for single_bbox in pred:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(0, 1.0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                conf_score = single_bbox[5].item()
                ax.text(
                    x0, y0 + 15, f"{obj_cls}, {conf_score:.2f}",
                    size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )
    # fmt: on
    plt.show()