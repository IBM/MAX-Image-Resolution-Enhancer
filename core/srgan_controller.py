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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from core.SRGAN.model import generator
from core.SRGAN.ops import deprocessLR, deprocess
import numpy as np
import skimage
import io


class SRGAN_controller:
    '''This class functions as a controller for the scripts contained in the SRGAN directory.'''

    def __init__(self, checkpoint, NUM_RESBLOCK=16):
        '''Initialize the TF graph.'''
        # Initialize the input with the correct dimensions
        self.inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')

        # Build the network
        with tf.variable_scope('generator'):
            gen_output = generator(self.inputs_raw, 3, reuse=False, is_training=False, num_resblock=NUM_RESBLOCK)

        with tf.name_scope('convert_image'):
            # Deprocess the model output images
            inputs = deprocessLR(self.inputs_raw)
            outputs = deprocess(gen_output)

            # Convert back to uint8
            converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
            converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

        with tf.name_scope('encode_image'):
            self.save_fetch = {
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
            }

        # Define the weight initializer (At inference time, we only need to restore the weight of the generator)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initializer = tf.train.Saver(var_list)

        # Define the initialization operation
        tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        # Load the pretrained model
        weight_initializer.restore(self.sess, checkpoint)

    def upscale(self, INPUT_IMAGE):
        '''Upscale an image with factor 4x.'''

        # Verify that the INPUT_IMAGE is indeed a np.array with the correct dtype
        if INPUT_IMAGE.dtype != np.float32:
            raise TypeError("Invalid type: %r" % INPUT_IMAGE.dtype)

        if INPUT_IMAGE.shape[0] != 1 or INPUT_IMAGE.shape[-1] != 3:
            raise ValueError(f"Invalid INPUT_IMAGE.shape: {INPUT_IMAGE.shape}")

        # Send the image through the network
        results = self.sess.run(self.save_fetch, feed_dict={self.inputs_raw: INPUT_IMAGE})

        # Convert the image bytestream to a skimage object
        output_image = skimage.io.imread(io.BytesIO(results['outputs'][0]), plugin='imageio')
        return output_image
