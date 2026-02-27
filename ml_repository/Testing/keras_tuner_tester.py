import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
from .base_tester import BaseTester
from Support.data_handler import DataHandler


class KerasTunerTester(BaseTester):
    """
    Tester for Keras Tuner models.
    """
    
    def __init__(self, config_handler):
        super().__init__(config_handler)
        self.data_handler = DataHandler(config_handler)
        
    def load_model(self, model_path):
        """
        Load the best model from Keras Tuner.
        """
        try:
            # For Keras Tuner, the model_path should be the tuner directory
            # Load the best model from the tuner
            if hasattr(self, 'tuner') and self.tuner is not None:
                self.model = self.tuner.get_best_models(1)[0]
            else:
                # Try to load directly if it's a saved model
                if model_path.endswith('.h5') or model_path.endswith('.keras'):
                    self.model = keras.models.load_model(model_path)
                else:
                    # Try loading as a saved model directory
                    self.model = keras.models.load_model(model_path)
            
            self.logger.info(f"Keras Tuner model loaded successfully from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading Keras Tuner model: {str(e)}")
            raise
    
    def prepare_test_data(self, test_data):
        """
        Prepare test data for Keras Tuner model prediction.
        """
        try:
            # test_data is already preprocessed (X, y) tuple from master.py
            if isinstance(test_data, (list, tuple)):
                features = test_data[0]
            else:
                features = test_data
            
            # Reshape if needed (for time series models)
            if len(features.shape) == 2:
                # Add batch dimension if needed
                if features.shape[1] == self.config.get('input_shape', [96])[0]:
                    features = features.reshape(-1, features.shape[1])
                else:
                    features = features.reshape(-1, 1)
            
            self.logger.info(f"Test data prepared. Shape: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing test data: {str(e)}")
            raise
    
    def predict(self, test_data):
        """
        Make predictions using the loaded Keras Tuner model.
        """
        try:
            # Make predictions
            predictions = self.model.predict(test_data, verbose=0)
            
            # Flatten predictions if needed
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            
            self.logger.info(f"Predictions made. Shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def extract_targets(self, test_data):
        """
        Extract target values from test data for Keras Tuner models.
        """
        try:
            # test_data is already preprocessed (X, y) tuple from master.py
            if isinstance(test_data, (list, tuple)) and len(test_data) > 1:
                targets = test_data[1]
            else:
                # If no targets in processed data, try to extract from original
                if hasattr(test_data, 'targets'):
                    targets = test_data.targets
                elif hasattr(test_data, 'y'):
                    targets = test_data.y
                else:
                    raise ValueError("Could not extract targets from test data")
            
            # Ensure targets are flattened
            if len(targets.shape) > 1:
                targets = targets.flatten()
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error extracting targets: {str(e)}")
            raise 