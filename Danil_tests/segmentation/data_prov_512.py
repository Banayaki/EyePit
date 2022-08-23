"""
DESCRIPTION OF THE AUGMENTATION.
"""
import numpy as np
import glob
import cv2
from tqdm import tqdm

from makiflow.augmentation import AffineAugment, ElasticAugment, ImageCutter, Data, FlipAugment


def normalize_data(X, Y):
    X = np.asarray(X).astype(np.float32) / 255
    X = [i for i in X]
    Y = np.asarray(Y).astype(np.uint8) // 10
    Y = [i for i in Y]
    
    return X, Y
    

def calc_num_pos(labels):
    area = labels[0].shape[0] * labels[0].shape[1]
    return [area - (label == 0).sum() for label in labels]
    
def get_test_data():
    Xtest, Ytest = [], []

    masks = glob.glob('../datasets/eyes/test_set/masks/*.bmp')
    masks.sort()
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('masks', 'images'))
        mask = cv2.imread(mask_name)

        img = cv2.resize(img, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, interpolation=cv2.INTER_NEAREST)

        Xtest.append(img)
        Ytest.append(mask[:, :, 0]) 
       
    return get_test_data_crop(Xtest, Ytest)


def get_test_data_crop(imgs,lbls):   
    
    cropped_images, cropped_masks, _ = ImageCutter.image_and_mask_cutter(imgs, lbls,
        window_h=512, window_w=512, step_x=250, step_y=250, scale_factor=0.01)
        
    class_power = {i: 0 for i in range(10)}
    for img in cropped_masks:
        u = np.unique(img)
        for uniq in u:
            class_power[uniq] += 1
    
    print(class_power, ' power of test set')
    return cropped_images, cropped_masks

def get_train_data():
    Xtrain, Ytrain = [], []
    # 1024 '/mnt/data/med_data/balanced_batches/bb_danil1/masks/*.bmp'
    # 512 /mnt/data/med_data/balanced_batches/bb_danil2_512
    masks = glob.glob('/mnt/data/med_data/balanced_batches/bb_danil2_512/masks/*.bmp')
    masks.sort()
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('masks', 'images'))
        mask = cv2.imread(mask_name)

        Xtrain.append(img)
        Ytrain.append(mask[:, :, 0])

    return Xtrain, Ytrain


def get_data():
    Xtrain, Ytrain = get_train_data()
    Xtest, Ytest = get_test_data()

    print(len(Xtrain), len(Ytrain))
    print(f'test is {len(Ytest)}') 

    data = Data(Xtrain, Ytrain)
    data = ElasticAugment(num_maps=4, std=8, noise_invert_scale=7, border_mode='reflect_101', keep_old_data=True)(data)
    data = FlipAugment([FlipAugment.FLIP_HORIZONTALLY, FlipAugment.FLIP_VERTICALLY], True)(data)

    Xtrain, Ytrain = data.get_data()
    Xtrain, _ = normalize_data(Xtrain, [])
    
    print(len(Xtrain))
    
    num_pos = calc_num_pos(Ytrain)
    Xtest, _ = normalize_data(Xtest, [])
    
    return Xtrain, Ytrain, num_pos, Xtest, Ytest