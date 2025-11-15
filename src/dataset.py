# src/dataset.py

import os
import glob
import numpy as np
import cv2


def load_lr_hr_pairs(lr_dir, hr_dir, max_images=None):
    """
    Load paired LR and HR images from two folders.

    lr_dir: directory with LR images
    hr_dir: directory with HR images
    max_images: optional limit on number of pairs

    Returns:
        lr_images: numpy array (N, H_lr, W_lr, 3) in [0, 1]
        hr_images: numpy array (N, H_hr, W_hr, 3) in [0, 1]
    """
    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.*")))
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.*")))

    if max_images is not None:
        lr_files = lr_files[:max_images]
        hr_files = hr_files[:max_images]

    lr_images = []
    hr_images = []

    for lr_path, hr_path in zip(lr_files, hr_files):
        img_lr = cv2.imread(lr_path)
        img_hr = cv2.imread(hr_path)

        if img_lr is None or img_hr is None:
            continue

        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

        lr_images.append(img_lr)
        hr_images.append(img_hr)

    lr_images = np.array(lr_images, dtype=np.float32) / 255.0
    hr_images = np.array(hr_images, dtype=np.float32) / 255.0

    return lr_images, hr_images
