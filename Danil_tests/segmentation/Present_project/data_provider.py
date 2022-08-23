"""
DESCRIPTION OF THE AUGMENTATION.
"""
import numpy as np
import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
from generator_provider import get_generator

def normalize_data(X, Y):
    X = np.asarray(X).astype(np.float32)
    X /= 255
    X = [i for i in X]
    Y = np.asarray(Y).astype(np.uint8)
    Y //=  10
    Y = [i for i in Y]
    
    return X, Y

def normalize_data_tf(X):
    ses = tf.Session()
    img = tf.placeholder(dtype=np.float32,name='img', shape=[None,1024,1024,3])
    div = img / tf.constant(255.0, dtype=tf.float32)
    
    answer = ses.run(div, feed_dict={img:X})
    
    X = [i for i in answer]
    
    ses.close()
    return X


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

        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        Xtest.append(img)
        Ytest.append(mask[:, :, 0])

    return Xtest, Ytest

def get_data_cv3():
    Xtest, Ytest = get_test_data(path='/raid/rustam/med_data/balanced_batches/batch_3/test_set_wo_4')
    print('Size of test set is ', len(Xtest))
    gen = get_generator(path_images='/raid/rustam/med_data/balanced_batches/batch_3/train_set/aug_set/set_1024_10k_wo_4_4/10k/images',
                        path_masks='/raid/rustam/med_data/balanced_batches/batch_3/train_set/aug_set/set_1024_10k_wo_4_4/10k/masks')
    
    Xtest, _ = normalize_data(Xtest, [])
    
    return Xtest, Ytest, gen