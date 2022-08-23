from makiflow.layers import *
from makiflow.models.segmentation.segmentator import Segmentator
from makiflow.save_recover import Builder
from makiflow.trainers import SegmentatorTrainer
from makiflow.metrics import categorical_dice_coeff, confusion_mat
import makiflow as mf

import tensorflow as tf
import numpy as np
import glob
import cv2
import seaborn as sns
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter


def load_data(path_to_data='/mnt/data/rustam/med_data/all_data'):
    Xtrain = []
    Ytrain = []

    masks = glob.glob(f'{path_to_data}/masks/*.bmp')
    i = 0
    masks.sort()
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('masks', 'images'))
        mask = cv2.imread(mask_name)
        print(f'{i}: {mask_name}')
        i += 1
        
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        Xtrain.append(img)
        Ytrain.append(mask[:, :, 0])
        
    return Xtrain, Ytrain

def normalize_data(Xtrain, Ytrain, Xtest, Ytest):
    Xtrain = np.asarray(Xtrain).astype(np.float32) / 255
    Xtrain = [i for i in Xtrain]
    Ytrain = np.asarray(Ytrain).astype(np.uint8) // 10
    Ytrain = [i for i in Ytrain]
    
    Xtest = np.asarray(Xtest, dtype=np.float32) / 255
    Xtest = [i for i in Xtest]
    Ytest = np.asarray(Ytest, dtype=np.uint8) // 10
    Ytest = [i for i in Ytest]
    return Xtrain, Ytrain, Xtest, Ytest


if __name__ == "__main__":
    mf.set_main_gpu(0)
    images, labels = load_data()
    images, labels, Xtest, Ytest = normalize_data(images, labels, [], [])
    model = Builder.segmentator_from_json('/home/rustam/EyePit/Models/x65/xception_unet_v13.json', batch_size=1)
    model.set_session(mf.get_low_memory_sess())
    model.load_weights('/home/rustam/EyePit/PaperGradientBalancing/Test#3_x-65_CrossValidation3Batches(Artem)_BalancedData/exp_cv1/x-65/MakiSegmentator_gamma=2_opt_name=adam1_bsz=8/epoch_19/weights.ckpt')
    for i in range(100):
        st = time()
        model.predict(images[0:1])
        print(time() - st)