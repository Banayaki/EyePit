
import makiflow
from makiflow.generators.gen_base import SegmentIterator, PathGenerator
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle

class SubCyclicGenerator(PathGenerator):
    def __init__(self, path_batches_images, path_batches_masks):
        """
        Generator for pipeline, which gives next element in sub-cyclic order
        Parameters
        ----------
        path_batches_masks : list
            A list of groups of paths to masks.
        path_batches_images : list
            A list of groups of paths to images.
        """
        assert (len(path_batches_images) == len(path_batches_masks))

        self.batches_images = path_batches_images
        self.batches_masks = path_batches_masks

        self.batches_images, self.batches_masks = shuffle(self.batches_images, self.batches_masks)

        for i in range(len(self.batches_masks)):
            self.batches_images[i], self.batches_masks[i] = shuffle(self.batches_images[i], self.batches_masks[i])

    def next_element(self):
        current_batch = 0
        counter_batches = [0 for _ in range(len(self.batches_images))]
        while True:
            if current_batch == (len(self.batches_images) - 1) and counter_batches[-1] == (
                    len(self.batches_images[-1]) - 1):
                self.batches_images, self.batches_masks = shuffle(self.batches_images, self.batches_masks)

                for i in range(len(self.batches_masks)):
                    self.batches_images[i], self.batches_masks[i] = shuffle(self.batches_images[i], self.batches_masks[i])

                current_batch = 0
                counter_batches = [0 for _ in range(len(self.batches_images))]

            el = {
                SegmentIterator.image: self.batches_images[current_batch][counter_batches[current_batch]],
                SegmentIterator.mask: self.batches_masks[current_batch][counter_batches[current_batch]]
            }

            counter_batches[current_batch] += 1
            current_batch = (current_batch + 1) % len(self.batches_images)

            yield el


from makiflow.models.segmentation.map_methods import LoadDataMethod, ComputePositivesPostMethod
from makiflow.models.segmentation.map_methods import NormalizePostMethod, RGB2BGRPostMethod, SqueezeMaskPostMethod

from makiflow.models.segmentation.gen_layers import InputGenLayer

def get_generator(path_images, path_masks):
    map_method = LoadDataMethod(image_shape=[1024, 1024, 3], mask_shape=[1024, 1024, 3])
    map_method = SqueezeMaskPostMethod()(map_method)
    map_method = RGB2BGRPostMethod()(map_method)
    map_method = NormalizePostMethod(use_float64=True)(map_method)
    map_method = ComputePositivesPostMethod()(map_method)
    return InputGenLayer(
        prefetch_size=8,
        batch_size=8, 
        path_generator=SubCyclicGenerator(path_images, path_masks),
        name='Input',
        map_operation=map_method,
        num_parallel_calls=5
    )
