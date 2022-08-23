"""
DESCRIPTION OF THE AUGMENTATION.
"""
import numpy as np
import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
from makiflow.augmentation import AffineAugment, ElasticAugment, ImageCutter, Data, FlipAugment
from generator_provider import get_generator

def normalize_data(X, Y):
    X = np.asarray(X).astype(np.float32)
    X /= 255
    X = [i.astype(np.float32) for i in X]
    Y = np.asarray(Y).astype(np.uint8)
    Y //=  10
    Y = [i for i in Y]
    
    return X, Y

def calc_num_pos(labels):
    area = labels[0].shape[0] * labels[0].shape[1]
    return [area - (label == 0).sum() for label in labels]
    
def get_test_data(path):
    Xtest, Ytest = [], []

    masks = glob.glob(path + '/masks/*.bmp')
    masks.sort()
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('masks', 'images'))
        mask = cv2.imread(mask_name)
        
        Xtest.append(img)
        Ytest.append(mask[:, :, 0])

        

    def get_test_data(imgs,lbls):   
    
        cropped_images, cropped_masks, _ = ImageCutter.image_and_mask_cutter(imgs, lbls,
            window_h=512, window_w=512, step_x=260, step_y=260, scale_factor=0.01)

        class_power = {i: 0 for i in range(10)}
        for img in cropped_masks:
            u = np.unique(img)
            for uniq in u:
                class_power[uniq] += 1

        print(class_power, ' power of test set')
        return cropped_images, cropped_masks
    
    return get_test_data(Xtest,Ytest)

def get_data_cv1():
    Xtest, Ytest = get_test_data(path='/mnt/data/med_data/balanced_batches/batch_1/test_set_wo_4')
    #/mnt/data/med_data/balanced_batches/batch_1/train_set/aug_set/set_1024_0
    gen = get_generator(path_images='/mnt/data/med_data/balanced_batches/batch_1/exp_5_512/images',
                        path_masks='/mnt/data/med_data/balanced_batches/batch_1/exp_5_512/masks')
    Xtest, _ = normalize_data(Xtest, [])
    
    return Xtest, Ytest, gen


def get_data_cv2():
    Xtest, Ytest = get_test_data(path='/mnt/data/med_data/balanced_batches/batch_2/test_set_wo_4')
    #
    gen = get_generator(path_images='/mnt/data/med_data/balanced_batches/batch_2/exp_5_512/images',
                        path_masks='/mnt/data/med_data/balanced_batches/batch_2/exp_5_512/masks')
    Xtest, _ = normalize_data(Xtest, [])

    return Xtest, Ytest, gen

def get_data_cv3():
    Xtest, Ytest = get_test_data(path='/mnt/data/med_data/balanced_batches/batch_3/test_set_wo_4')
    #
    gen = get_generator(path_images='/mnt/data/med_data/balanced_batches/batch_3/exp_5_512/images',
                        path_masks='/mnt/data/med_data/balanced_batches/batch_3/exp_5_512/masks')
    
    Xtest, _ = normalize_data(Xtest, [])
    
    return Xtest, Ytest, gen                      