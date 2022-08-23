from makiflow.models.ssd.tools.data_preparator_v2 import DataPreparatorV2 as DataPreparator
from makiflow.models.ssd.tools.data_preparing import prepare_data_rcnn
from makiflow.models.ssd.ssd_utils import prepare_data_v2 as prepare_data

from makiflow.generators.ssd.data_preparation import record_mp_od_train_data, record_od_train_data

from makiflow.metrics.od_utils import parse_dicts

from makiflow.augmentation.object_detection.augment_ops import FlipAugment, ContrastBrightnessAugment, GaussianBlur, GaussianNoiseAugment, FlipType
from makiflow.augmentation.object_detection.data_provider import Data

import tensorflow as tf
import makiflow as mf

mf.set_main_gpu(0)
tf.enable_eager_execution()

import numpy as np
from tqdm import tqdm
import json
import os

"""
    type_aug : int
        0 --- FLIP_HORIZONTALLY 
        1 --- FLIP_VERTICALLY 
        2 --- FLIP_BOTH
        3 --- ContrastBrightnessAugment
        4 --- GaussianBlur
        5 --- GaussianNoiseAugment

    type_dboxes : int
        0 --- dummy style
        1 --- rcnn like
        
    norm_images : str
        'norm' --- x / 255
        'coffee' --- x / 128 - 1
"""



def create_tfrecord_data(
    path_to_dboxes_xy,
    path_train_set,
    path_classes,
    path_save_numpy_stuff,
    postfix_name_to_save,
    path_save_tfrecords,
    file_name,
    path_to_dboxes_wh=None,
    iou_threshold=0.3,
    path_to_data='/mnt/data/voc2012/VOCdevkit/VOC2012/JPEGImages/', 
    use_aug=False,
    type_aug=[0],
    use_dark=False,
    keep_old_data=False,
    type_dboxes=0,
    norm_images='norm'):
    dboxes_xy = np.load(path_to_dboxes_xy)
    if path_to_dboxes_wh is not None:
        dboxes_wh = np.load(path_to_dboxes_wh)

    with open(path_train_set, 'r') as fp:
        train_set = json.load(fp)


    with open(path_classes, 'r') as f:
        class2name = json.load(f)

    # prepare data
    preparator = DataPreparator(train_set, class2name, path_to_data)
    preparator.load_images()
    preparator.resize_images_and_bboxes((300, 300))
    if use_aug:
        images = preparator.get_images()
        bboxes = preparator.get_bboxes()
        _, labels = parse_dicts(train_set, class2name)

        flip = None
        if 0 in type_aug:
            flip = FlipType.FLIP_HORIZONTALLY
        elif 1 in type_aug:
            flip = FlipType.FLIP_VERTICALLY
        elif 2 in type_aug:
            flip = FlipType.FLIP_BOTH

        data = Data(images, bboxes, labels)
        if flip is not None:
            data = FlipAugment(flip_type_list=[flip], keep_old_data=keep_old_data)(data)

        if 3 in type_aug:
            if not use_dark:
                data = ContrastBrightnessAugment(params=[(1.07, .09)], keep_old_data=keep_old_data)(data)
            else:
                data = ContrastBrightnessAugment(params=[(0.93, -.09)], keep_old_data=keep_old_data)(data)
        if 4 in type_aug:
            data = GaussianBlur(keep_old_data=keep_old_data)(data)
            
        if 5 in type_aug:
            data = GaussianNoiseAugment(noise_tensors_num=10, std=8., keep_old_data=keep_old_data)(data)

        new_images, new_bboxes, new_labels = data.get_data()

        labels = []
        loc_masks = []
        locs = []
        for b, l in tqdm(zip(new_bboxes, new_labels)):
            if type_dboxes == 0:
                loc_masks_single, labels_single, locs_single = prepare_data(b,l, dboxes_xy, iou_threshold)
            elif type_dboxes == 1:
                assert(dboxes_wh is None, 'dboxes_wh is None')
                loc_masks_single, labels_single, locs_single = prepare_data_rcnn(b,l,dboxes_wh, dboxes_xy, iou_threshold)
            labels.append(labels_single)
            loc_masks.append(loc_masks_single)
            locs.append(locs_single)

        labels = np.asarray(labels).astype(np.int32)
        loc_masks = np.asarray(loc_masks).astype(np.float32)
        locs = np.asarray(locs).astype(np.float32)
        
        if norm_images == 'norm':
            images = (np.asarray(new_images) / 255).astype(np.float32)
        elif norm_images == 'coffee':
            images = ((np.asarray(new_images) / 128) - 1).astype(np.float32)
    else:
        if type_dboxes == 0:
            loc_masks, labels, locs = preparator.generate_masks_labels_locs(dboxes_xy, iou_threshold=iou_threshold)
        elif type_dboxes == 1:
            assert(dboxes_wh is None, 'dboxes_wh is None')
            loc_masks, labels, locs = preparator.generate_masks_labels_offsets_rcnn(dboxes_xy, dboxes_wh, iou_threshold=iou_threshold)
        images = preparator.get_images()
        
        if norm_images == 'norm':
            images = (np.asarray(images) / 255).astype(np.float32)
        elif norm_images == 'coffee':
            images = ((np.asarray(images) / 128) - 1).astype(np.float32)

    np.save(path_save_numpy_stuff + f'/masks_iou{iou_threshold}_{postfix_name_to_save}.npy', loc_masks)
    np.save(path_save_numpy_stuff + f'/labels_iou{iou_threshold}_{postfix_name_to_save}.npy', labels)
    np.save(path_save_numpy_stuff + f'/locs_iou{iou_threshold}_{postfix_name_to_save}.npy', locs)


    loc_masks = np.load(path_save_numpy_stuff + f'/masks_iou{iou_threshold}_{postfix_name_to_save}.npy')
    labels = np.load(path_save_numpy_stuff + f'/labels_iou{iou_threshold}_{postfix_name_to_save}.npy')
    locs = np.load(path_save_numpy_stuff + f'/locs_iou{iou_threshold}_{postfix_name_to_save}.npy')

    os.mkdir(path_save_tfrecords + f'/{file_name}')

    record_mp_od_train_data(images, loc_masks, locs,
                            labels, path_save_tfrecords + f'/{file_name}' + '/record', 320)

