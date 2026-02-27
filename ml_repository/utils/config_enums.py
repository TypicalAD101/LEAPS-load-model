from enum import Enum

class ModelPaths(Enum):
    KERAS = 'model_keras'
    KERAS_TUNER = 'model_keras_tuner'
    DARTS = 'model_darts'

class DatabaseNames(Enum):
    KERAS = 'keras_val_db'
    KERAS_TUNER = 'keras_tuner_val_db'
    DARTS = 'darts_val_db'

class TrainingParams(Enum):
    BATCH_SIZE = 'batch_size'
    EPOCHS = 'epochs'
    VALIDATION_SPLIT = 'validation_split'
    SEQUENCE_LENGTH = 'sequence_length' 