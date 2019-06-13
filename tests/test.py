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

import pytest
import requests
import io
from PIL import Image


def test_swagger():
    '''Test the Swagger UI'''

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Image Resolution Enhancer'


def test_metadata():
    '''Test the metadata of the model.'''

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'max-image-resolution-enhancer'
    assert metadata['name'] == 'Super-Resolution Generative Adversarial Network (SRGAN)'
    assert metadata['description'] == 'SRGAN trained on the OpenImagesV4 dataset.'
    assert metadata['license'] == 'Apache V2'


def call_model(file_path):
    '''Send an input image through the network.'''
    model_endpoint = 'http://localhost:5000/model/predict'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/png')}
        r = requests.post(url=model_endpoint, files=file_form)
        assert r.status_code == 200
        im = Image.open(io.BytesIO(r.content))
        return im


def test_predict():
    '''Check the prediction output of 5 test images.'''

    # Test the output image of the woman
    im = call_model(file_path='assets/test_examples/low_resolution/woman.png')
    assert im.size == (424, 636)

    # Test the output image of the astronaut
    im = call_model(file_path='assets/test_examples/low_resolution/astronaut.png')
    assert im.size == (1276, 1380)

    # Test the output image of the food
    im = call_model(file_path='assets/test_examples/low_resolution/food.png')
    assert im.size == (512, 320)

    # Test the output image of the palm tree
    im = call_model(file_path='assets/test_examples/low_resolution/palm_tree.png')
    assert im.size == (948, 1412)

    # Test the output image of the elephant
    im = call_model(file_path='assets/test_examples/low_resolution/elephant.png')
    assert im.size == (868, 1392)


if __name__ == '__main__':
    pytest.main([__file__])
