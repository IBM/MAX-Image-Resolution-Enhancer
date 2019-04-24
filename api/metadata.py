from core.model import ModelWrapper
from maxfw.core import MAX_API, MetadataAPI, METADATA_SCHEMA


class ModelMetadataAPI(MetadataAPI):

    @MAX_API.marshal_with(METADATA_SCHEMA)
    def get(self):
        """Return the metadata associated with the model"""
        return ModelWrapper.MODEL_META_DATA
