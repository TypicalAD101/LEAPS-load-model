import numpy as np
from darts import TimeSeries
from darts.models import TCNModel, RNNModel, BlockRNNModel
import pickle
import os
from .base_tester import BaseTester
from Support.data_handler import DataHandler


class DartsTester(BaseTester):
    """
    Tester for Darts models.
    """
    
    def __init__(self, config_handler):
        super().__init__(config_handler)
        self.data_handler = DataHandler(config_handler)
        
    def load_model(self, model_path):
        """
        Load a trained Darts model.
        """
        try:
            from darts.models import NBEATSModel  # or the correct Darts model class
            self.model = NBEATSModel.load(model_path)
            self.logger.info(f"Darts model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading Darts model: {str(e)}")
            raise
    
    def prepare_test_data(self, test_data):
        """
        Prepare test data for Darts model prediction.
        """
        try:
            # test_data is already a TimeSeries from master.py
            if isinstance(test_data, TimeSeries):
                return test_data
            else:
                # If not a TimeSeries, try to convert
                if isinstance(test_data, (list, tuple)):
                    features = test_data[0]
                else:
                    features = test_data
                
                # Convert to TimeSeries
                return TimeSeries.from_values(features)
            
        except Exception as e:
            self.logger.error(f"Error preparing test data: {str(e)}")
            raise
    
    def predict(self, test_data, train_series=None):
        """
        Make multi-step predictions using the loaded Darts model.
        If train_series is provided, use its last 96 points as history to predict the full test horizon.
        """
        try:
            from darts import TimeSeries
            seq_length = self.config.get('input_chunk_length', 96)
            forecast_horizon = len(test_data)
            if train_series is not None:
                if not isinstance(train_series, TimeSeries):
                    train_series = TimeSeries.from_values(train_series)
                historical_input = train_series[-seq_length:]
            else:
                # Fallback: use the first seq_length points of test_data as history (not ideal)
                historical_input = test_data[:seq_length]
                test_data = test_data[seq_length:]
                forecast_horizon = len(test_data)
            predictions = self.model.predict(n=forecast_horizon, series=historical_input)
            if isinstance(predictions, TimeSeries):
                predictions = predictions.values().flatten()
            return predictions
        except Exception as e:
            self.logger.error(f"Error making multi-step predictions: {str(e)}")
            raise

    def test(self, test_data, model_path, train_series=None):
        """
        Test the Darts model using multi-step forecasting.
        """
        try:
            self.logger.info("Starting model testing...")
            self.load_model(model_path)
            if self.model is None:
                raise ValueError("Failed to load model")
            prepared_test_data = self.prepare_test_data(test_data)
            # If train_series is not provided, try to get it from DataHandler
            if train_series is None and hasattr(self.data_handler, 'preprocess_data'):
                train_series = self.data_handler.preprocess_data('train')
            predictions = self.predict(prepared_test_data, train_series=train_series)
            # The actuals are the test_data values
            if isinstance(prepared_test_data, TimeSeries):
                actuals = prepared_test_data.values().flatten()
            else:
                actuals = prepared_test_data
            metrics = self.calculate_metrics(predictions, actuals)
            self.print_test_results(metrics, predictions, actuals)
            self.test_predictions = predictions
            self.test_actuals = actuals
            self.logger.info("Testing completed successfully!")
            return metrics, predictions, actuals
        except Exception as e:
            self.logger.error(f"Error during testing: {str(e)}")
            raise 