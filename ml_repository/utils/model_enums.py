from enum import Enum

class ModelType(Enum):
    KERAS = 'k'
    KERAS_TUNER = 't'
    DARTS = 'd'
    ARIMA = 'a'

class ConfigKeys(Enum):
    TRAIN_MODE = 'train_mode'
    VAL_MODE = 'val_mode'
    IMPL_MODE = 'impl_mode'
    SEQUENCE_LENGTH = 'sequence_length'

MODEL_TYPE_TO_PATH_KEY = {
    'k': 'model_keras',
    'kt': 'model_keras_tuner',
    'd': 'model_darts',
    'a': 'model_arima'
} 