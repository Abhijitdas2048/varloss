# imports
import numpy as np
from PIL import Image

import keras.backend as K
from data_loader import load_data
from albumentations import (
    Compose, HorizontalFlip, Rotate, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, JpegCompression,
    ToFloat, ShiftScaleRotate
)
import cv2

AUGMENTATIONS_TEST = Compose([
    CLAHE(p=1.0, clip_limit=2.0),
    ToFloat(max_value=255)
])

# DataGenerator class
class DataGenerator(object):
    def __init__(self,labels,variance, target_shape=(64, 64), rescale=1.0, batch_size=1):
        # Total number of data samples. It actually depends on data, e.g.
        # number of files or number of images in dataset. For this example
        # I manually set this to 100.
        self._total = 10000
        
        # Target data size, generally equals to input shape of the network
        # without the channel axis.
        self._shape = target_shape
        
        # Scaling factor to be multiplied with each sample, e.g. for image
        # data typically 1/255 is used.
        self._scale = rescale
        
        # Batch size.
        self._batch = batch_size
        
        # Number of batches.
        self._steps = int(self._total / self._batch + 0.5)
        
        # Initial index. This is used to track batches.
        self._index = 0

        self._idx = 0

        self._labels = labels
        self._variance = variance
        # Seed for reproducibility
        #np.random.seed(seed)
    
    def flow(self):
        while True:
            # Here we will return data as 2-tuple. First element is inputs to
            # network and second element is target. For multi input network we
            # simply make the first element a list itself that will contain
            # all the inputs.
            x1 = []
            y1 = []
            y2 = []
            (train_x, train_y), (test_x, test_y) = load_data()
            # End point to track batch.
            endidx = self._index + self._batch
            
            # Index rolling. Just to be safe roll back to zero if index exceeds
            # max length. We are not actually using indexing in this simple
            # example. But it will be required in actual practice.
            self._index = endidx if endidx < self._total else 0
            
            # Let's populate the lists with some dummy data now.
            for _ in range(self._batch):
                if self._idx < 10000:
                    image_1 = test_x[self._idx, :,:]
                    label = self._labels[self._idx, :]
                    var = self._variance[self._idx, :]
                    self._idx = self._idx + 1
                    x1.append(image_1)
                    y2.append(var)
                    y1.append(label)
                    
            if self._idx >=10000:
                    self._idx = 0
            
            # Convert lists into arrays and apply scaling.
            x2 = np.stack([AUGMENTATIONS_TEST(image=x)["image"] for x in x1], axis=0)
            #x2 = np.asarray(x2, np.float32) 
            y1 = np.asarray(y1, np.float32)
            y2 = np.asarray(y2, np.float32)
            
            # This is generator so we have to use `yield` rather than `return`.
            yield [[x2, y2], y1]
