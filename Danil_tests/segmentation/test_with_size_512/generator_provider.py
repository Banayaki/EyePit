
import makiflow
from makiflow.generators.gen_base import SegmentIterator, PathGenerator
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle


class Generator(PathGenerator):
    def __init__(self, path_images, path_masks):
        self.images = glob(os.path.join(path_images, '*.bmp'))
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


from makiflow.models.segmentation.map_methods import LoadDataMethod, ComputePositivesPostMethod
from makiflow.models.segmentation.map_methods import NormalizePostMethod, RGB2BGRPostMethod, SqueezeMaskPostMethod

from makiflow.models.segmentation.gen_layers import InputGenLayer

def get_generator(path_images, path_masks):
    map_method = LoadDataMethod(image_shape=[512, 512, 3], mask_shape=[512, 512, 3])
    map_method = SqueezeMaskPostMethod()(map_method)
    map_method = RGB2BGRPostMethod()(map_method)
    map_method = NormalizePostMethod(use_float64=True)(map_method)
    map_method = ComputePositivesPostMethod()(map_method)
    return InputGenLayer(
        prefetch_size=8,
        batch_size=8, 
        path_generator=Generator(path_images, path_masks),
        name='Input',
        map_operation=map_method,
        num_parallel_calls=5
    )
