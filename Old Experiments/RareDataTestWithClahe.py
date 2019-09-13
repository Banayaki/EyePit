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


def use_clahe(img, clahe_kernel=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))):
    for i in range(3):
        img[:, :, i] = clahe_kernel.apply(img[:, :, i])
    return img


def load_data(path_to_data='../dataset/mask'):
    Xtrain = []
    Ytrain = []

    masks = glob.glob(f'{path_to_data}/*.bmp')
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('mask', 'imgs'))
        mask = cv2.imread(mask_name)
        Xtrain.append(use_clahe(img))
        Ytrain.append(mask)
        
    return Xtrain, Ytrain


def calc_num_pos(labels):
    area = labels[0].shape[0] * labels[0].shape[1]
    return [area - (label == 0).sum() for label in labels]


if __name__ == "__main__":
    # Load data
    images, labels = load_data()
    rare_images = []
    rare_labels = []
    
    for i, (img, lbl) in enumerate(zip(images, labels)):
        u = np.unique(lbl)
        if 40 in u or 70 in u or 80 in u:
            rare_images.append(img)
            rare_labels.append(lbl)
            
    images = rare_images
    labels = rare_labels
    
    Xtest = []
    Ytest = []
    Xtrain = []
    Ytrain = []

    for i, label in enumerate(labels):
        uniq = np.unique(label)
        if i in [7, 11, 75, 99]:
            Xtest.append(images[i])
            Ytest.append(label)
            continue
        else:
            Xtrain.append(images[i])
            Ytrain.append(label)
        
    Xtrain, Ytrain, _ = ImageCutter.image_and_mask_cutter(Xtrain, Ytrain, 384, 384, 128, 128, 0.005)
    Xtest, Ytest, _ = ImageCutter.image_and_mask_cutter(Xtest, Ytest, 384, 384, 128, 128, 0.005)
    
    Xtrain = np.asarray(Xtrain).astype(np.float32) / 255
    Xtrain = [i for i in Xtrain]
    Ytrain = np.asarray(Ytrain).astype(np.uint8) // 10
    Ytrain = [i for i in Ytrain]
    
    Xtest = np.asarray(Xtest).astype(np.float32) / 255
    Xtest = [i for i in Xtest]
    Ytest = np.asarray(Ytest).astype(np.uint8) // 10
    Ytest = [i for i in Ytest]
    
    data = Data(Xtrain, Ytrain)
    data = AffineAugment(num_matrices=5, noise_type='gaussian', keep_old_data=True)(data)
    data = ElasticAugment(num_maps=6, border_mode='reflect_101',  keep_old_data=True)(data)
    
    aug_images, aug_labels = data.get_data()
    print(len(aug_images))
    
    aug_labels = [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in aug_labels]
    Ytest = [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in Ytest]
    num_pos = calc_num_pos(aug_labels)
    
    assert(sum([i > 0 for i in num_pos]) == len(num_pos))
    
    # Start test
    mf.set_main_gpu(0)
    
    trainer = SegmentatorTrainer('experiment.json', 'clahep')
    trainer.set_test_data(Xtest, Ytest)
    trainer.set_train_data(aug_images, aug_labels, num_pos)
    
    trainer.start_experiments()
    print('Done')