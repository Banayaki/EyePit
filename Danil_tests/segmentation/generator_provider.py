import sys
sys.path.append('/home/rustam/EyePit/Danil_tests/MakiFlow/')
import makiflow
from makiflow.models.segmentation.gen_base import PathGenerator, SegmentIterator

images_path = '/mnt/data/med_data/pipeline_data/images/'
masks_path = '/mnt/data/med_data/pipeline_data/masks/'
#/mnt/data/med_data/balanced_batches/bb_danil1/masks/*.bmp
#'/mnt/data/med_data/pipeline_data/images/'
#/mnt/data/med_data/pipeline_data_5k
#/mnt/data/med_data/pipeline_data_8k
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle



class Generator(PathGenerator):
    def __init__(self, path_imgs, path_masks):
        self.images = glob(os.path.join(path_imgs, '*.bmp'))
        self.masks = glob(os.path.join(path_masks, '*.bmp'))
        
    def next_element(self):
        index = 0
        while True:
            if index % len(self.images) == 0:
                self.images, self.masks = shuffle(self.images, self.masks)
                index = 0
            #index = np.random.randint(low=0, high=len(self.images))
            el = {
                SegmentIterator.image: self.images[index],
                SegmentIterator.mask: self.masks[index]
            }
            index += 1
            
            yield el


from makiflow.models.segmentation.map_methods import LoadDataMethod, ResizePostMethod, ComputePositivesPostMethod, \
    SqueezeMaskPostMethod, NormalizePostMethod

from makiflow.models.segmentation.gen_layers import InputGenLayer

def get_generator():
    map_method = LoadDataMethod(image_shape=[1024, 1024, 3], mask_shape=[1024, 1024])
    #map_method = NormalizePostMethod()(map_method)
    #map_method = SqueezeMaskPostMethod()(map_method)
    map_method = ComputePositivesPostMethod()(map_method)
    return InputGenLayer(
        prefetch_size=6,
        batch_size=6, 
        path_generator=Generator(images_path, masks_path),
        name='Input',
        map_operation=map_method,
        num_parallel_calls=5
    )
