
import makiflow
from makiflow.generators.gen_base import SegmentIterator, PathGenerator
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle



from makiflow.generators.segmentation.pathgenerator import CyclicGeneratorSegment, RandomGeneratorSegment, SubCyclicGeneratorSegment


from makiflow.generators.segmentation.map_methods import LoadDataMethod, ComputePositivesPostMethod
from makiflow.generators.segmentation.map_methods import NormalizePostMethod, RGB2BGRPostMethod, SqueezeMaskPostMethod

from makiflow.generators.segmentation.gen_layers import InputGenLayer

def get_generator(path_images, path_masks):
    map_method = LoadDataMethod(image_shape=[1024, 1024, 3], mask_shape=[1024, 1024, 3])
    map_method = SqueezeMaskPostMethod()(map_method)
    map_method = RGB2BGRPostMethod()(map_method)
    map_method = NormalizePostMethod(use_float64=True)(map_method)
    map_method = ComputePositivesPostMethod()(map_method)
    return InputGenLayer(
        prefetch_size=4,
        batch_size=4, 
        path_generator=CyclicGeneratorSegment(path_images, path_masks),
        name='Input',
        map_operation=map_method,
        num_parallel_calls=5
    )