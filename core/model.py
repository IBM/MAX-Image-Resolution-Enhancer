#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import io
import logging

import numpy as np
import skimage.color
import skimage.io
import skimage.transform

from flask import abort
from maxfw.model import MAXModelWrapper
from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta
from core.srgan_controller import SRGAN_controller

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):
    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        # Initialize the SRGAN controller
        self.SRGAN = SRGAN_controller(checkpoint=DEFAULT_MODEL_PATH)

        logger.info('Loaded model')

    def _read_image(self, image_data):
        '''Read the image from a Bytestream.'''
        image = skimage.io.imread(io.BytesIO(image_data), plugin='imageio')
        return image

    def _pre_process(self, image):
        '''
        Preprocess the image.

        1. Verify the dimensions
        2. Resize if we exceed the maximum dimensions permitted by our model
        3. Normalize the image
        4. Convert to standardized input format
        '''
        # Standardize input dtype of image
        image = image.astype('uint8')

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Resize dimensions that are too large to 500px (instead of raising an error)
        logger.info(f'image input dim: {image.shape[0]}x{image.shape[1]}')

        # a. find factor
        factor = np.ceil(max(image.shape[0], image.shape[1]) / 500)

        # b. resize
        if factor > 1:  # if at least one image dimension is bigger than 500px
            if factor > 4:
                message = "The dimensions of the image are too big (>2000px). The image would have been downscaled instead."
                logger.error(message)
                abort(400, message)

            image = skimage.transform.resize(image,
                                             (np.floor(image.shape[0] / factor), np.floor(image.shape[1] / factor)),
                                             anti_aliasing=True)
            logger.info(f'image resized to: {image.shape[0]}x{image.shape[1]}')

        # Normalize image
        image = image / np.max(image)

        # Convert the image to numpy array with dtype float32 as required by the SRGAN
        # (1, H, W, C)
        image = np.array([image]).astype(np.float32)

        return image

    def _predict(self, image):
        '''Call the model'''
        return self.SRGAN.upscale(image)

    def write_image(self, image):
        '''Return the generated image as output.'''
        logger.info(f'image output dim: {image.shape[0]}x{image.shape[1]}')
        stream = io.BytesIO()
        skimage.io.imsave(stream, image)
        stream.seek(0)
        return stream

    def _post_process(self, result):
        '''Post-processing.'''
        return result
