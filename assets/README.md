# Asset Details

## Model files

The final SRGAN model was trained on 600k images from the [OpenImages V4](https://storage.googleapis.com/openimages/web/index.html) dataset. The weights are released here under the [Apache2.0](https://www.apache.org/licenses/LICENSE-2.0) license found in the root of this repository.

_Note: the finetuned model files are hosted on [IBM Cloud Object Storage](http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/max-image-super-resolution-generator/v1.0.0/assets.tar.gz)._

## Test Examples (assets/testexamples)

This directory contains to subdirectories:
* low_resolution
* original

The low resolution images can be used as input for the model, and the resulting high resolution images can be compared with the 'original' images to measure performance.

The source of the images is the following:

**[OpenImages V4](https://storage.googleapis.com/openimages/web/index.html)**

_[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) licensed_

- `monkey_and_man.png`

**[Pexels](https://www.pexels.com/royalty-free-images/)**

_[CC0](https://creativecommons.org/publicdomain/zero/1.0/) licensed_

- `airplane.png`
- `astronaut.png`
- `elephant.png`
- `face.png`
- `face_paint.png`
- `food.png`
- `palm_tree.png`
- `penguin.png`
- `woman.png`