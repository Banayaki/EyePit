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

def get_train_data():
    Xtrain, Ytrain = [], []
    #/mnt/data/med_data/balanced_batches/bb_danil1
    #/mnt/data/med_data/pipeline_data_8k
    masks = glob.glob('/mnt/data/med_data/pipeline_data_5k/masks/*.bmp')
    masks.sort()
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('masks', 'images'))
        mask = cv2.imread(mask_name)

        Xtrain.append(img)
        Ytrain.append(mask[:, :, 0])

    return Xtrain, Ytrain

def poor_dividive1(images, divide=9):
    answer_images = None
    part = len(images) // divide
    ses = tf.Session()
    inp = tf.placeholder(dtype=np.float32,name='inp', shape=[part,1024,1024,3])
    answer = inp / 255.0
    for i in range(divide):
        print(i)
        if answer_images is None:
            answer_images = ses.run(answer, feed_dict = {inp:images[i*part:part*(i+1)]})
        else:
            answer_images = np.vstack((ses.run(answer, feed_dict = {inp:images[i*part:part*(i+1)]}), answer_images))
    ses.close()
    return answer_images

def poor_dividive(images, divide):
    answer_images = []
    part = len(images) // divide
    ses = tf.Session()
    inp = tf.placeholder(dtype=np.float32,name='inp', shape=[None,1024,1024,3])
    answer = inp / 255
    for i in range(part):
        if i == part - 1:
            answer_images.append(ses.run(answer, feed_dict = {inp:images[i*part:]}).astype(np.float32))
        elif answer_images is None:
            answer_images.append(ses.run(answer, feed_dict = {inp:images[i*part:part*(i+1)]}).astype(np.float32))
            continue
        answer_images.append(ses.run(answer, feed_dict = {inp:images[i*part:part*(i+1)]}).astype(np.float32))
        
     
    return answer_images

def get_data_cv1():
    Xtest, Ytest = get_test_data(path='/mnt/data/med_data/balanced_batches/batch_1/exp_2/test_set/only_7_class')
    
    gen = get_generator(path_images='/mnt/data/med_data/balanced_batches/batch_1/exp_2/train_set/only_7_class/images',
                        path_masks='/mnt/data/med_data/balanced_batches/batch_1/exp_2/train_set/only_7_class/masks')
    Xtest, _ = normalize_data(Xtest, [])
    
    return Xtest, Ytest, gen


def get_data_cv2():
    Xtest, Ytest = get_test_data(path='/mnt/data/med_data/balanced_batches/batch_2/exp_2/test_set/only_7_class')
    
    gen = get_generator(path_images='/mnt/data/med_data/balanced_batches/batch_2/exp_2/train_set/only_7_class/images',
                        path_masks='/mnt/data/med_data/balanced_batches/batch_2/exp_2/train_set/only_7_class/masks')
    Xtest, _ = normalize_data(Xtest, [])

    return Xtest, Ytest, gen

def get_data_cv3():
    Xtest, Ytest = get_test_data(path='/mnt/data/med_data/balanced_batches/batch_3/exp_2/test_set/only_7_class')
    
    gen = get_generator(path_images='/mnt/data/med_data/balanced_batches/batch_3/exp_2/train_set/only_7_class/images',
                        path_masks='/mnt/data/med_data/balanced_batches/batch_3/exp_2/train_set/only_7_class/masks')
    
    Xtest, _ = normalize_data(Xtest, [])
    
    return Xtest, Ytest, gen


def get_data():
    #Xtrain, Ytrain = get_train_data()
    Xtest, Ytest = get_test_data()
    #Xtest = np.asarray(Xtest).astype(np.float32)
    #Xtest = [i for i in Xtest]
    
    Ytest = np.asarray(Ytest).astype(np.int32)
    Ytest = [i for i in Ytest]
    
    #print(len(Xtrain), len(Ytrain))
    print(f'test is {len(Ytest)}') 

    #data = Data(Xtrain, Ytrain)
    #data = ElasticAugment(num_maps=4, std=8, noise_invert_scale=7, border_mode='reflect_101', keep_old_data=True)(data)
    #data = FlipAugment([FlipAugment.FLIP_HORIZONTALLY, FlipAugment.FLIP_VERTICALLY], True)(data)

    #Xtrain, Ytrain = data.get_data()
    #Xtrain, _ = normalize_data(Xtrain, [])
    #Xtest = normalize_data_tf(Xtest)
    #Xtrain = poor_dividive1(Xtrain)
    
    #print(len(Xtrain))
    #print(Xtrain[0].shape)
    #Xtrain = [i for i in Xtrain]
    #num_pos = calc_num_pos(Ytrain)
    #Xtest, _ = normalize_data(Xtest, [])
    Xtest = normalize_data_tf(Xtest)
    Xtest = [i for i in Xtest]
    #return Xtrain, Ytrain, num_pos, Xtest, Ytest
    return Xtest, Ytest