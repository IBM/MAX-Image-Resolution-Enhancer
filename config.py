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

# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'

# API metadata
API_TITLE = 'MAX Image Resolution Enhancer'
API_DESC = 'Upscale low-resolution images by a factor of 4. This model was trained on the OpenImagesV4 dataset.'
API_VERSION = '1.1.0'

# default model
MODEL_NAME = 'SRGAN'
DEFAULT_MODEL_PATH = 'assets/SRGAN/model'

MODEL_META_DATA = {
    'id': 'max-image-resolution-enhancer',
    'name': 'Super-Resolution Generative Adversarial Network (SRGAN)',
    'description': 'SRGAN trained on the OpenImagesV4 dataset.',
    'type': 'Image-To-Image Translation Or Transformation',
    'source': 'https://developer.ibm.com/exchanges/models/all/max-image-resolution-enhancer/',
    'license': 'Apache V2'
}
