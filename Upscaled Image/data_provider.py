"""
DESCRIPTION OF THE AUGMENTATION.
"""
import numpy as np
import glob
import cv2
from tqdm import tqdm

from makiflow.augmentation import AffineAugment, ElasticAugment, ImageCutter, Data


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

                if overlap_x:
                    for dy in range(int((current_height - window_h) / step_y)):
                        crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                            img,
                            mask,
                            dy * step_y, dy * step_y + window_h, current_width - window_w, current_width)
                        if ImageCutter.has_class(crop_mask, classes_to_get):
                            cropped_images.append(crop_img)
                            cropped_masks.append(crop_mask)

                if overlap_x and overlap_y:
                    crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                        img,
                        mask,
                        current_height - window_h, current_height, current_width - window_w, current_width)
                    if ImageCutter.has_class(crop_mask, classes_to_get):
                        cropped_images.append(crop_img)
                        cropped_masks.append(crop_mask)

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


def normalize_data(Xtrain, Ytrain, Xtest, Ytest):
    Xtrain = np.asarray(Xtrain).astype(np.float32) / 255
    Xtrain = [i for i in Xtrain]
    Ytrain = np.asarray(Ytrain).astype(np.uint8) // 10
    Ytrain = [i for i in Ytrain]
    
    Xtest = np.asarray(Xtest).astype(np.float32) / 255
    Xtest = [i for i in Xtest]
    Ytest = np.asarray(Ytest).astype(np.uint8) // 10
    Ytest = [i for i in Ytest]
    return Xtrain, Ytrain, Xtest, Ytest
    

def calc_num_pos(labels):
    area = labels[0].shape[0] * labels[0].shape[1]
    return [area - (label == 0).sum() for label in labels]


def load_data(path_to_data='../dataset/mask'):
    Xtrain = []
    Ytrain = []

    masks = glob.glob(f'{path_to_data}/*.bmp')
    for mask_name in tqdm(masks):
        img = cv2.imread(mask_name.replace('mask', 'imgs'))
        mask = cv2.imread(mask_name)
        
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        
        Xtrain.append(img)
        Ytrain.append(mask[:, :, 0])
        
    return Xtrain, Ytrain
    
def get_test_data():
    images, labels = load_data()
    imgs, lbls = [], []
    for i, (img, lbl) in enumerate(zip(images, labels)):
        if i in [2, 24, 41, 85, 75, 6, 7]:
            imgs += [img]
            lbls += [lbl]
            
    cropped_images1, cropped_masks1, _ = ImageCutter.image_and_mask_cutter(imgs, lbls,
        window_h=256, window_w=256, step_x=50, step_y=50, classes_to_get=[40, 70, 80])
    
    cropped_images2, cropped_masks2, _ = ImageCutter.image_and_mask_cutter(
        imgs, lbls, window_h=256, window_w=256, step_x=40, step_y=40, classes_to_get=[70])
    
    cropped_images3, cropped_masks3, _ = ImageCutter.image_and_mask_cutter(
        imgs, lbls, window_h=256, window_w=256, step_x=60, step_y=60, classes_to_get=[60])
    
    cropped_images4, cropped_masks4, _ = ImageCutter.image_and_mask_cutter(
        imgs, lbls, window_h=256, window_w=256, step_x=60, step_y=60, classes_to_get=[10])
        
    cropped_images = cropped_images1 + cropped_images2 + cropped_images3 + cropped_images4 + cropped_images1[:8]
    cropped_masks = cropped_masks1 + cropped_masks2 + cropped_masks3 + cropped_masks4 + cropped_masks1[:8]
    print(len(cropped_images))
    assert(len(cropped_images) % 8 == 0)
    return cropped_images, cropped_masks

def get_train_data():
    images, labels = load_data()
    imgs, lbls = [], []
    for i, (img, lbl) in enumerate(zip(images, labels)):
        if i in [2, 24, 41, 85, 75, 6, 7]:
            continue
        imgs += [img]
        lbls += [lbl]
            
    cropped_images1, cropped_masks1, _ = ImageCutter.image_and_mask_cutter(imgs, lbls,
        window_h=256, window_w=256, step_x=50, step_y=50, classes_to_get=[40, 70, 80])
    
    cropped_images2, cropped_masks2, _ = ImageCutter.image_and_mask_cutter(
        imgs, lbls, window_h=256, window_w=256, step_x=40, step_y=40, classes_to_get=[70])
    
    cropped_images3, cropped_masks3, _ = ImageCutter.image_and_mask_cutter(
        imgs, lbls, window_h=256, window_w=256, step_x=60, step_y=60, classes_to_get=[60])
    
    cropped_images4, cropped_masks4, _ = ImageCutter.image_and_mask_cutter(
        imgs, lbls, window_h=256, window_w=256, step_x=60, step_y=60, classes_to_get=[10])
        
    cropped_images = cropped_images1 + cropped_images2 + cropped_images3 + cropped_images4
    cropped_masks = cropped_masks1 + cropped_masks2 + cropped_masks3 + cropped_masks4
    return cropped_images, cropped_masks


# USE ONLY ABSOLUTE PATHS TO DATA TO MAKE THE FILE REUSABLE
def get_data():
    images, labels = load_data()
    
    Xtest, Ytest = get_test_data()
    Xtrain, Ytrain = get_train_data()
    
    data = Data(Xtrain, Ytrain)
    data = ElasticAugment(num_maps=3, border_mode='reflect_101',  keep_old_data=True)(data)
    
    Xtrain, Ytrain = data.get_data()
    print(len(Xtrain), len(Ytrain))
    
    num_pos = calc_num_pos(Ytrain)
    Xtrain, Ytrain, Xtest, Ytest = normalize_data(Xtrain, Ytrain, Xtest, Ytest)
    
    return Xtrain, Ytrain, num_pos, Xtest, Ytest
