import pandas as pd
from Support.data_handler import DataHandler
from Support.config_handler import ConfigHandler
from utils.model_enums import ModelType
from Training.keras_model_train import keras_model_train as kt_t
from Training.keras_tuner_model import keras_tuner_model as tm_t
from Training.darts_model import darts_model as dm_t
from Training.arima_model import arima_model as arima_t

from Validation.keras_validator import KerasValidator
from Validation.keras_tuner_validator import KerasTunerValidator
from Validation.darts_validator import DartsValidator
from Validation.arima_validator import ArimaValidator

from Testing.keras_tester import KerasTester
from Testing.keras_tuner_tester import KerasTunerTester
from Testing.darts_tester import DartsTester
from Testing.arima_tester import ArimaTester

import os
import numpy as np
from utils.logger import init_logger, get_logger

logger = init_logger()

class Train:
    def __init__(self):
        # Initialize config handler
        self.config_handler = ConfigHandler()
        self.model_config = self.config_handler.get_model_config()
        self.paths = self.config_handler.get_paths()
        self.db_config = self.config_handler.get_database_config()
        self.training_config = self.config_handler.get_training_config()

        # Initialize handlers
        self.dh = DataHandler()
        self.kt_t = kt_t()
        self.tm_t = tm_t()
        self.dm_t = dm_t()
        self.arima_t = arima_t()

        # Initialize validators with logger
        self.kt_v = KerasValidator(logger=logger)
        self.tm_v = KerasTunerValidator(logger=logger)
        self.dm_v = DartsValidator(logger=logger)
        self.arima_v = ArimaValidator(logger=logger)

        # Initialize testers
        self.kt_tester = KerasTester(self.config_handler)
        self.tm_tester = KerasTunerTester(self.config_handler)
        self.dm_tester = DartsTester(self.config_handler)
        self.arima_tester = ArimaTester(logger=logger)

        logger.info('Train pipeline initialized.')
        self.run()

    def run(self):
        logger.info('Pipeline run started.')
        # self.data_handling()
        # self.model_training(self.model_config['train_mode'])
        # self.model_validation(self.model_config['val_mode'])
        # self.model_testing(self.model_config['train_mode'])
        self.model_implementation(self.model_config['impl_mode'])
        # self.post_processing()
        # self.gif_visualization()
        logger.info('Pipeline run completed.')

    def data_handling(self):
        logger.info('Starting data handling...')
        try:
            self.dh.data_loading()
            # self.dh.data_visualization()
            logger.info('Data handling completed.')
        except Exception as e:
            logger.error('Error during data handling', exc_info=True)
            raise

    def model_training(self, training_mode):
        logger.info(f'Starting model training: mode={training_mode}')
        seq_length = self.model_config['sequence_length']
        try:
            if training_mode == ModelType.KERAS.value:
                X_train, y_train = self.dh.preprocess_data('train', seq_length)
                X_val, y_val = self.dh.preprocess_data('validation', seq_length)
                logger.debug('Keras training data preprocessed.')
                model_save_path = self.paths[ModelType.KERAS.value]
                self.kt_t.train(X_train, y_train, seq_length=seq_length, model_save_path=model_save_path, **self.training_config)
                logger.info('Keras model training completed.')

            elif training_mode == ModelType.KERAS_TUNER.value:
                X_train, y_train = self.dh.preprocess_data('train', seq_length)
                X_val, y_val = self.dh.preprocess_data('validation', seq_length)
                logger.debug('Keras Tuner training data preprocessed.')
                model_save_path = self.paths[ModelType.KERAS_TUNER.value]
                self.tm_t.train(seq_length, X_train, X_val, y_train, y_val, model_save_path=model_save_path, **self.training_config)
                logger.info('Keras Tuner model training completed.')

            elif training_mode == ModelType.DARTS.value:
                train_series = self.dh.preprocess_data('train')
                val_series = self.dh.preprocess_data('validation')
                logger.debug('DARTS training data preprocessed.')
                model_save_path = self.paths[ModelType.DARTS.value]
                self.dm_t.train(train_series, val_series, model_save_path=model_save_path, **self.training_config)
                logger.info('DARTS model training completed.')

            elif training_mode == ModelType.ARIMA.value:
                model_save_path = self.paths[ModelType.ARIMA.value]
                self.arima_t.train(model_save_path=model_save_path)
                logger.info('ARIMA model training completed.')
        except Exception as e:
            logger.error('Error during model training', exc_info=True)
            raise

    def model_validation(self, validation_mode):
        logger.info(f'Starting model validation: mode={validation_mode}')
        seq_length = self.model_config['sequence_length']
        try:
            if validation_mode == ModelType.KERAS.value:
                model_path = self.paths[ModelType.KERAS.value]
                X_val, y_val = self.dh.preprocess_data('validation', seq_length)
                self.kt_v.run_validation(model_path, (X_val, y_val), seq_length, print_results=True)
                logger.info('Keras model validation completed.')

            elif validation_mode == ModelType.KERAS_TUNER.value:
                model_path = self.paths[ModelType.KERAS_TUNER.value]
                X_val, y_val = self.dh.preprocess_data('validation', seq_length)
                self.tm_v.run_validation(model_path, (X_val, y_val), seq_length, print_results=True)
                logger.info('Keras Tuner model validation completed.')

            elif validation_mode == ModelType.DARTS.value:
                model_path = self.paths[ModelType.DARTS.value]
                val_series = self.dh.preprocess_data('validation')
                self.dm_v.run_validation(model_path, val_series, val_series, seq_length, print_results=True)
                logger.info('DARTS model validation completed.')

            elif validation_mode == ModelType.ARIMA.value:
                model_path = self.paths[ModelType.ARIMA.value]
                self.arima_v.run_validation(model_path, seq_length)
                logger.info('ARIMA model validation completed.')
        except Exception as e:
            logger.error('Error during model validation', exc_info=True)
            raise

    def model_testing(self, testing_mode):
        logger.info(f'Starting model testing: mode={testing_mode}')
        seq_length = self.model_config['sequence_length']
        try:
            if testing_mode == ModelType.KERAS.value:
                model_path = self.paths[ModelType.KERAS.value]
                test_data = self.dh.preprocess_data('test')
                self.kt_tester.test(test_data, model_path)
                logger.info('Keras model testing completed.')

            elif testing_mode == ModelType.KERAS_TUNER.value:
                model_path = self.paths[ModelType.KERAS_TUNER.value]
                test_data = self.dh.preprocess_data('test')
                self.tm_tester.test(test_data, model_path)
                logger.info('Keras Tuner model testing completed.')

            elif testing_mode == ModelType.DARTS.value:
                model_path = self.paths[ModelType.DARTS.value]
                test_data = self.dh.preprocess_data('test')
                self.dm_tester.test(test_data, model_path)
                logger.info('DARTS model testing completed.')

            elif testing_mode == ModelType.ARIMA.value:
                model_path = self.paths[ModelType.ARIMA.value]
                self.arima_tester.test(model_path, seq_length)
                logger.info('ARIMA model testing completed.')
        except Exception as e:
            logger.error('Error during model testing', exc_info=True)
            raise

    def model_implementation(self, implementation_mode):
        logger.info(f'Starting model implementation: mode={implementation_mode}')
        try:
            from Implementation.implementation import run_all_models_implementation
            run_all_models_implementation()
            logger.info('All models implementation completed.')
        except Exception as e:
            logger.error('Error during model implementation', exc_info=True)
            raise

    def post_processing(self):
        logger.info('Starting post-processing...')
        try:
            self.dh.postprocess_data()
            logger.info('Post-processing completed.')
        except Exception as e:
            logger.error('Error during post-processing', exc_info=True)
            raise

    def gif_visualization(self):
        logger.info('Starting GIF visualization...')
        try:
            self.dh.gif_visualization()
            logger.info('GIF visualization completed.')
        except Exception as e:
            logger.error('Error during GIF visualization', exc_info=True)
            raise


if __name__ == '__main__':
    Train()
