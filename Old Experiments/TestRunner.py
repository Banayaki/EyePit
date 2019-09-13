from makiflow.layers import *
from makiflow.models.segmentation.segmentator import Segmentator
from makiflow.augmentation import AffineAugment, ElasticAugment, ImageCutter, Data
from makiflow.trainers import SegmentatorTrainer
import makiflow as mf

import tensorflow as tf
import numpy as np
import glob
import cv2
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter
from copy import copy

import cv2


class ImageCutter:

    @staticmethod
    def image_and_mask_cutter(
        images, masks, window_h, window_w, step_x, step_y, classes_to_get,
        use_all_px=True
    ):
        """
        Crops `images` and `masks` using sliding window with resize.
        Parameters
        ----------
        images : list
            List of input images.
        masks : list
            List of input masks.
        window_h : int
            Output image height.
        window_w : int
            Output image width.
        step_x : int
            Sliding window step by OX.
        step_y : int
            Sliding window step by OX.
        scale_factor : float
            Scale factor, must be in range (0, 1). After each 'sliding window step' the original images
            are resized to (previous_width * scale_factor, previous_height * scale_factor).
        postprocessing : func
            Post processing function, using on cropped image (may be function what calculate num positives pixels).
        use_all_px : bool
            If True, all pixels of image would be in output lists.

        Returns
        -------
        Three list:
            1. cropped images
            2. cropped masks
            3. additional list (result of post processing)
        """
        assert (0 < scale_factor < 1)
        assert (len(images) > 0)
        assert (len(images) == len(masks))
        assert (window_h > 0 and window_w > 0 and step_x > 0 and step_y > 0)

        cropped_images = []
        cropped_masks = []
        additional_list = []
        dx = 0
        dy = 0

        for index, (img, mask) in enumerate(zip(images, masks)):
            print(index)
            assert (img.shape[:2] == mask.shape[:2])
            current_height, current_width = img.shape[:2]


            for dy in range(int((current_height - window_h) / step_y)):
                for dx in range(int((current_width - window_w) / step_x)):
                    crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                        img,
                        mask,
                        dy * step_y, dy * step_y + window_h, dx * step_x, dx * step_x + window_w)
                    if ImageCutter.has_class(crop_mask, classes_to_get):
                        cropped_images.append(crop_img)
                        cropped_masks.append(crop_mask)

            if use_all_px:
                overlap_y = dy * step_y + window_h != current_height
                overlap_x = dx * step_x + window_w != current_width
                if overlap_y:
                    for dx in range(int((current_width - window_w) / step_x)):
                        crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                            img,
                            mask,
                            current_height - window_h, current_height, dx * step_x, dx * step_x + window_w)
                        if ImageCutter.has_class(crop_mask, classes_to_get):
                            cropped_images.append(crop_img)
                            cropped_masks.append(crop_mask)

                        if postprocessing is not None:
                            additional_list.append(postprocessing(crop_img, crop_mask))
                if overlap_x:
                    for dy in range(int((current_height - window_h) / step_y)):
                        crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                            img,
                            mask,
                            dy * step_y, dy * step_y + window_h, current_width - window_w, current_width)
                        if ImageCutter.has_class(crop_mask, classes_to_get):
                            cropped_images.append(crop_img)
                            cropped_masks.append(crop_mask)

                        if postprocessing is not None:
                            additional_list.append(postprocessing(crop_img, crop_mask))
                if overlap_x and overlap_y:
                    crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                        img,
                        mask,
                        current_height - window_h, current_height, current_width - window_w, current_width)
                    if ImageCutter.has_class(crop_mask, classes_to_get):
                        cropped_images.append(crop_img)
                        cropped_masks.append(crop_mask)

                    if postprocessing is not None:
                        additional_list.append(postprocessing(crop_img, crop_mask))

        return cropped_images, cropped_masks, additional_list

    @staticmethod
    def crop_img_and_mask(img, mask, up, down, left, right):
        crop_img = img[up: down, left: right]
        crop_mask = mask[up: down, left: right]
        return crop_img, crop_mask
    
    @staticmethod
    def has_class(mask, needed):
        actual = np.unique(mask)
        for need in needed:
            if need in actual:
                return True
        return False

def load_data(path_to_data='../dataset/mask'):
    Xtrain = []
    Ytrain = []

    masks = glob.glob(f'{path_to_data}/*.bmp')
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('mask', 'imgs'))
        mask = cv2.imread(mask_name)
        Xtrain.append(img)
        Ytrain.append(mask)
        
    return Xtrain, Ytrain


def calc_num_pos(labels):
    area = labels[0].shape[0] * labels[0].shape[1]
    return [area - (label == 0).sum() for label in labels]


if __name__ == "__main__":
    # Load data
    images, labels = load_data()
    
    
    
    Xtest = []
    Ytest = []
    Xtrain = []
    Ytrain = []

    for i, label in enumerate(labels):
        uniq = np.unique(label)
        if i in [25, 41, 14, 19, 43, 2, 5]:
            Xtest.append(images[i])
            Ytest.append(label)
            continue
        else:
            Xtrain.append(images[i])
            Ytrain.append(label)
            
    for image, label in zip(copy(Xtrain), copy(Ytrain)):
        uniq = np.unique(label)
        if 40 in uniq or 70 in uniq or 80 in uniq:
            Ytrain += [label] * 10
            Xtrain += [image] * 10
        
    Xtrain, Ytrain, _ = ImageCutter.image_and_mask_cutter(Xtrain, Ytrain, 256, 256, 128, 128, 0.05)
    Xtest, Ytest, _ = ImageCutter.image_and_mask_cutter(Xtest, Ytest, 256, 256, 128, 128, 0.05)
    
    Xtrain = np.asarray(Xtrain).astype(np.float32) / 255
    Xtrain = [i for i in Xtrain]
    Ytrain = np.asarray(Ytrain).astype(np.uint8) // 10
    Ytrain = [i for i in Ytrain]
    
    Xtest = np.asarray(Xtest).astype(np.float32) / 255
    Xtest = [i for i in Xtest]
    Ytest = np.asarray(Ytest).astype(np.uint8) // 10
    Ytest = [i for i in Ytest]
    
    data = Data(Xtrain, Ytrain)
    data = AffineAugment(num_matrices=1, noise_type='gaussian', keep_old_data=True)(data)
    data = ElasticAugment(num_maps=1, border_mode='reflect_101',  keep_old_data=True)(data)
    
    aug_images, aug_labels = data.get_data()
    print(len(aug_images))
    
    aug_labels = [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in aug_labels]
    Ytest = [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in Ytest]
    num_pos = calc_num_pos(aug_labels)
    
    assert(sum([i > 0 for i in num_pos]) == len(num_pos))
    
    # Start test
    mf.set_main_gpu(1)
    
    trainer = SegmentatorTrainer('experiment.json', 'resnet50')
    trainer.set_test_data(Xtest, Ytest)
    trainer.set_train_data(aug_images, aug_labels, num_pos)
    
    trainer.start_experiments()
    print('Done')