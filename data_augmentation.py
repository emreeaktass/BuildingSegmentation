import albumentations as A
import numpy as np
from glob import glob
import cv2

hor_flip = A.Compose([
    A.HorizontalFlip(p=0.5),
])

ver_flip = A.Compose([
    A.VerticalFlip(p=0.5),
])

crop_pad_128 = A.Compose([
    A.CropAndPad(px=128, percent=None, keep_size=True, p=0.5),
])

crop_pad_64 = A.Compose([
    A.CropAndPad(px=64, percent=None, keep_size=True, p=0.5),
])

crop_random = A.Compose([
    A.RandomResizedCrop(width=256, height=256, p=0.5),
])

rotate = A.Compose([
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=(-15, 15), p=0.5),
])

shift = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=0.5),
])

gauss_noise = A.Compose([
    A.GaussNoise(var_limit=(10, 100), p=0.5),
])

gauss_blur = A.Compose([
    A.GaussianBlur(p=0.5),
])

brightness = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
])

clahe = A.Compose([
    A.CLAHE(p=0.5),
])

random_rotate_90 = A.Compose([
    A.RandomRotate90(p=0.5),
])

motion_blur = A.Compose([
    A.MotionBlur(p=0.5),
])

all_in_one = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.CropAndPad(px=256, percent=None, keep_size=True, p=0.3),
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=(-15, 15), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=0.3),
    A.GaussNoise(var_limit=(10, 100), p=0.3),
    A.GaussianBlur(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.3),
    A.RandomRotate90(p=0.3),
    A.MotionBlur(p=0.3),
])

def augmentation_():
    path_data = 'dataset/train/data/'
    path_mask = 'dataset/train/mask/'
    file_names_data = glob(path_data + '*')
    file_names_mask = glob(path_mask + '*')
    file_names_data.sort()
    file_names_mask.sort()

    for i, j in zip(file_names_data, file_names_mask):
        data = np.load(i)
        mask = np.load(j)

        old_sum = data.sum()

        transformed = all_in_one(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_all_in_one', transformed_image)
            np.save(j.split('.')[0] + '_all_in_one', transformed_mask)

        transformed = brightness(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_brightness', transformed_image)
            np.save(j.split('.')[0] + '_brightness', transformed_mask)

        transformed = gauss_blur(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_gauss_blur', transformed_image)
            np.save(j.split('.')[0] + '_gauss_blur', transformed_mask)

        transformed = gauss_noise(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_gauss_noise', transformed_image)
            np.save(j.split('.')[0] + '_gauss_noise', transformed_mask)

        transformed = shift(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_shift', transformed_image)
            np.save(j.split('.')[0] + '_shift', transformed_mask)

        transformed = rotate(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_rotate', transformed_image)
            np.save(j.split('.')[0] + '_rotate', transformed_mask)

        transformed = crop_random(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_crop_random', transformed_image)
            np.save(j.split('.')[0] + '_crop_random', transformed_mask)

        transformed = crop_pad_128(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_crop_pad_128', transformed_image)
            np.save(j.split('.')[0] + '_crop_pad_128', transformed_mask)

        transformed = crop_pad_64(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_crop_pad_64', transformed_image)
            np.save(j.split('.')[0] + '_crop_pad_64', transformed_mask)

        transformed = ver_flip(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_ver_flip', transformed_image)
            np.save(j.split('.')[0] + '_ver_flip', transformed_mask)

        transformed = hor_flip(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_hor_flip', transformed_image)
            np.save(j.split('.')[0] + '_hor_flip', transformed_mask)

        transformed = clahe(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_clahe', transformed_image)
            np.save(j.split('.')[0] + '_clahe', transformed_mask)

        transformed = random_rotate_90(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_random_rotate_90', transformed_image)
            np.save(j.split('.')[0] + '_random_rotate_90', transformed_mask)

        transformed = motion_blur(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        new_sum = transformed_image.sum()
        if new_sum != old_sum:
            np.save(i.split('.')[0] + '_motion_blur', transformed_image)
            np.save(j.split('.')[0] + '_motion_blur', transformed_mask)