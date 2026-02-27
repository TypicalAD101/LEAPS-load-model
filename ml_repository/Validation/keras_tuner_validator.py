import logging
from Validation.base_validator import BaseValidator
import tensorflow as tf
import numpy as np

class KerasTunerValidator(BaseValidator):
    """
    Validator for Keras Tuner models.
    Inherits from BaseValidator and implements model-specific logic for Keras Tuner.
    """
    def run_validation(self, model_path, test_data, actuals, seq_length=96, print_results=False):
        super().run_validation(model_path, test_data, actuals, seq_length, print_results)

    def _load_model(self, model_path):
        self.logger.info(f"Loading Keras Tuner model from {model_path}")
        return tf.keras.models.load_model(model_path)

    def _predict_one_step(self, model, input_data):
        print(f"[DEBUG] input_data type: {type(input_data)}, shape: {getattr(input_data, 'shape', None)}")
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        elif input_data.shape == (96, 1):
            input_data = input_data.T  # Transpose from (96, 1) to (1, 96)
        print(f"[DEBUG] reshaped input_data shape: {input_data.shape}")
        pred = model.predict(input_data)
        return pred[0][0]

    def _get_actual_value(self, actuals, i, seq_length):
        # For Keras Tuner, actuals are indexed by i + seq_length
        try:
            return actuals[i + seq_length]
        except IndexError:
            return actuals[-1] 