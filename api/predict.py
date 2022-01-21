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

from flask import send_file
from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI
from flask_restx import abort
from werkzeug.datastructures import FileStorage

# Set up parser for input data (http://flask-restx.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.parser()
input_parser.add_argument('image', type=FileStorage, location='files',
                          required=True,
                          help="An image file (RGB/HWC).\n"
                               "The ideal image size is 500x500 or smaller, "
                               "and the best results are observed using a PNG image.\n"
                               "Images with dimensions over 2000x2000 are not accepted and will result in an error.")


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    def post(self):
        """Make a prediction given input data"""

        args = input_parser.parse_args()
        try:
            input_data = args['image'].read()
            image = self.model_wrapper._read_image(input_data)
        except ValueError:
            abort(400,
                  "Please submit a valid image in PNG, Tiff or JPEG format")

        output_image = self.model_wrapper.predict(image)
        return send_file(self.model_wrapper.write_image(output_image), mimetype='image/png', attachment_filename='result.png')
