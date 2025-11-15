# src/preprocess_patches.py

import os
import cv2


def make_patches(
    img_dir,
    out_dir,
    patch_size=128,
    stride=58,
    max_images = None,
):
    """
    Create HR patches from large microscopy images (no augmentation).

    img_dir: directory with original images
    out_dir: directory to save HR patches
    patch_size: size of square patch (e.g., 128)
    stride: sliding window step size (e.g., 58)
    max_images: limit number of images for quick testing
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(img_dir))
    if max_images is not None:
        files = files[:max_images]

    for img_name in files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        ny = (h - patch_size) // stride + 1
        nx = (w - patch_size) // stride + 1

        base = os.path.splitext(img_name)[0]

        for j in range(nx):
            for i in range(ny):
                y0 = i * stride
                x0 = j * stride
                patch = img[y0:y0 + patch_size, x0:x0 + patch_size, :]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue
                out_name = f"{base}_{i:02d}_{j:02d}.jpg"
                cv2.imwrite(os.path.join(out_dir, out_name), patch)


def make_patches_with_aug(
    img_dir,
    out_dir,
    patch_size=128,
    stride=60,
    max_images=None,
):
    """
    Create HR patches from large images with augmentation:
    - original
    - 90, 180, 270 degree rotations
    - vertically flipped + same rotations
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(img_dir))
    if max_images is not None:
        files = files[:max_images]

    for img_name in files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        ny = (h - patch_size) // stride + 1
        nx = (w - patch_size) // stride + 1

        base = os.path.splitext(img_name)[0]

        for j in range(nx):
            for i in range(ny):
                y0 = i * stride
                x0 = j * stride
                patch = img[y0:y0 + patch_size, x0:x0 + patch_size, :]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue

                # Upper (original orientation) variants
                im1 = patch
                im2 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
                im3 = cv2.rotate(im1, cv2.ROTATE_180)
                im4 = cv2.rotate(im1, cv2.ROTATE_90_COUNTERCLOCKWISE)

                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_up.jpg"), im1)
                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_up_90.jpg"), im2)
                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_up_180.jpg"), im3)
                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_up_270.jpg"), im4)

                # Lower (vertically flipped) variants
                flip_v = cv2.flip(im1, 0)
                f2 = cv2.rotate(flip_v, cv2.ROTATE_90_CLOCKWISE)
                f3 = cv2.rotate(flip_v, cv2.ROTATE_180)
                f4 = cv2.rotate(flip_v, cv2.ROTATE_90_COUNTERCLOCKWISE)

                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_down.jpg"), flip_v)
                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_down_90.jpg"), f2)
                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_down_180.jpg"), f3)
                cv2.imwrite(os.path.join(out_dir, f"{base}_{i:02d}_{j:02d}_down_270.jpg"), f4)


def make_lr_from_hr(hr_dir, lr_dir, lr_size=(32, 32)):
    """
    Create LR patches (e.g., 32x32) by resizing HR patches (e.g., 128x128).

    hr_dir: directory with HR patches
    lr_dir: directory to save LR patches
    lr_size: output size of LR images
    """
    os.makedirs(lr_dir, exist_ok=True)
    files = sorted(os.listdir(hr_dir))

    for img_name in files:
        img_path = os.path.join(hr_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        lr_img = cv2.resize(img, lr_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_dir, img_name), lr_img)


if __name__ == "__main__":
    # Adjust these paths if needed
    raw_dir = "data/raw"
    patches_hr_dir = "data/patches_hr"
    patches_lr_dir = "data/patches_lr32"

    # 1) Create HR patches with augmentation
    make_patches_with_aug(
        img_dir=raw_dir,
        out_dir=patches_hr_dir,
        patch_size=128,
        stride=58,
        max_images=None,  # e.g., 10 for quick testing
    )

    # 2) Create LR patches from HR patches (e.g., 32x32)
    make_lr_from_hr(
        hr_dir=patches_hr_dir,
        lr_dir=patches_lr_dir,
        lr_size=(32, 32),
    )
