# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'

# API metadata
API_TITLE = 'MAX Image Resolution Enhancer'
API_DESC = 'Upscale low-resolution images by a factor of 4. This model was trained on the OpenImagesV4 dataset.'
API_VERSION = '1.0.0'

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
