class GPT2GeoException(Exception):
    """Base exception class for GPT-2Geo."""


class DatasetError(GPT2GeoException):
    """Exception raised for dataset-related errors."""
    def __init__(self, message="Error in dataset handling"):
        self.message = message
        super().__init__(self.message)


class ModelError(GPT2GeoException):
    """Exception raised for GPT-2 model-related errors."""
    def __init__(self, message="Error in GPT-2 model"):
        self.message = message
        super().__init__(self.message)
