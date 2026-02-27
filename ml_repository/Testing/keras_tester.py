import numpy as np
import tensorflow as tf
from tensorflow import keras
from .base_tester import BaseTester
from Support.data_handler import DataHandler


class KerasTester(BaseTester):
    """
    Tester for Keras models.
    """
    
    def __init__(self, config_handler):
        super().__init__(config_handler)
        self.data_handler = DataHandler(config_handler)
        
    def load_model(self, model_path):
        """
        Load a trained Keras model.
        """
        try:
            if model_path.endswith('.h5') or model_path.endswith('.keras'):
                self.model = keras.models.load_model(model_path)
            else:
                # Try loading as a saved model directory
                self.model = keras.models.load_model(model_path)
            
            self.logger.info(f"Keras model loaded successfully from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading Keras model: {str(e)}")
            raise
    
    def prepare_test_data(self, test_data):
        """
        Prepare test data for Keras model prediction.
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
        Make predictions using the loaded Keras model.
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
        Extract target values from test data for Keras models.
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
            
            # Ensure targets are flattened and converted to scalar values
            if len(targets.shape) > 1:
                targets = targets.flatten()
            
            # Convert numpy arrays to list of scalar values
            targets_list = []
            for t in targets:
                if hasattr(t, '__iter__') and not isinstance(t, str):
                    targets_list.append(float(t[0]) if len(t) > 0 else 0.0)
                else:
                    targets_list.append(float(t))
            
            return targets_list
            
        except Exception as e:
            self.logger.error(f"Error extracting targets: {str(e)}")
            raise 

    def recursive_forecast(self, X, y, seq_length=96, scaler=None, print_results=False):
        import numpy as np
        num_windows = X.shape[0] // seq_length
        all_preds = []
        all_actuals = []
        for i in range(num_windows):
            start_idx = i * seq_length
            end_idx = start_idx + seq_length
            if end_idx + seq_length > X.shape[0]:
                break
            # Initial history window
            history = X[start_idx].reshape(1, seq_length, 1)
            # True future values (normalized)
            true_future = []
            for j in range(seq_length):
                true_future.append(y[start_idx + j])
            # Recursive forecast
            preds = []
            current_window = history.copy()
            for step in range(seq_length):
                pred = self.model.predict(current_window, verbose=0)
                preds.append(pred[0, 0])
                # Update window: drop oldest, append new pred
                new_window = np.append(current_window[0, 1:, 0], pred[0, 0])
                current_window = new_window.reshape(1, seq_length, 1)
            # Denormalize if scaler is provided
            if scaler is not None:
                preds_denorm = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                true_denorm = scaler.inverse_transform(np.array(true_future).reshape(-1, 1)).flatten()
            else:
                preds_denorm = np.array(preds)
                true_denorm = np.array(true_future)
            all_preds.extend(preds_denorm)
            all_actuals.extend(true_denorm)
            if print_results:
                print(f"\nWindow {i+1} - Recursive 96-step forecast:")
                for k in range(seq_length):
                    print(f"Step {k+1}: Actual: {true_denorm[k]:.6f}, Predicted: {preds_denorm[k]:.6f}, Abs Diff: {abs(true_denorm[k] - preds_denorm[k]):.6f}")
        return np.array(all_preds), np.array(all_actuals)

    def test(self, test_data, model_path, scaler=None, print_results=False):
        """
        Main testing method that orchestrates the recursive multi-step testing process.
        """
        try:
            self.logger.info("Starting model testing (recursive 96-step forecast)...")
            # Load the model
            self.logger.info(f"Loading model from {model_path}")
            self.load_model(model_path)
            if self.model is None:
                raise ValueError("Failed to load model")
            # Prepare test data
            self.logger.info("Preparing test data...")
            prepared_data = self.prepare_test_data(test_data)
            # Extract actual values (assuming test_data contains targets)
            if isinstance(test_data, (list, tuple)) and len(test_data) > 1:
                targets = test_data[1]
            else:
                targets = self.extract_targets(test_data)
            # Run recursive forecast
            predictions, actuals = self.recursive_forecast(prepared_data, targets, seq_length=self.config.get('sequence_length', 96), scaler=scaler, print_results=print_results)
            # Calculate metrics
            self.logger.info("Calculating metrics...")
            metrics = self.calculate_metrics(predictions, actuals)
            # Print results
            self.print_test_results(metrics, predictions, actuals)
            # Store results for potential further use
            self.test_predictions = predictions
            self.test_actuals = actuals
            self.logger.info("Testing completed successfully!")
            return metrics, predictions, actuals
        except Exception as e:
            self.logger.error(f"Error during testing: {str(e)}")
            raise 

    def test_detailed(self, scaler=None, print_results=False):
        """
        Detailed test: window-by-window recursive forecasting with per-window metrics and results.
        Assumes test data is prepared using the 'train_max' split.
        """
        self.logger.info("Starting detailed window-by-window testing...")
        # Get test data using train_max split
        (_, _), (X_test, y_test) = self.data_handler.preprocess_data('train_max', seq_length=self.config.get('sequence_length', 96))
        seq_length = self.config.get('sequence_length', 96)
        num_test_windows = self.config.get('num_test_windows', 3)
        steps_per_day = seq_length
        all_window_results = []
        for i in range(num_test_windows):
            start_idx = i * steps_per_day
            # History window
            history = X_test[start_idx].reshape(1, seq_length, 1)
            # True future values
            true_future = [y_test[start_idx + j][0] for j in range(seq_length)]
            # Recursive forecast
            preds = []
            current_window = history.copy()
            for step in range(seq_length):
                pred = self.model.predict(current_window, verbose=0)
                preds.append(pred[0, 0])
                new_window = np.append(current_window[0, 1:, 0], pred[0, 0])
                current_window = new_window.reshape(1, seq_length, 1)
            # Denormalize if scaler is provided
            if scaler is not None:
                preds_denorm = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                true_denorm = scaler.inverse_transform(np.array(true_future).reshape(-1, 1)).flatten()
            else:
                preds_denorm = np.array(preds)
                true_denorm = np.array(true_future)
            errors = true_denorm - preds_denorm
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            window_result = {
                'window_index': i,
                'actual_values': true_denorm.tolist(),
                'predicted_values': preds_denorm.tolist(),
                'errors': errors.tolist(),
                'metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
            }
            all_window_results.append(window_result)
            if print_results:
                print(f"\nWindow {i+1} - Recursive {seq_length}-step forecast:")
                for k in range(seq_length):
                    print(f"Step {k+1}: Actual: {true_denorm[k]:.6f}, Predicted: {preds_denorm[k]:.6f}, Abs Diff: {abs(errors[k]):.6f}")
                print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        # Overall metrics
        all_preds = np.concatenate([w['predicted_values'] for w in all_window_results])
        all_actuals = np.concatenate([w['actual_values'] for w in all_window_results])
        overall_mae = np.mean(np.abs(all_actuals - all_preds))
        overall_rmse = np.sqrt(np.mean((all_actuals - all_preds) ** 2))
        overall_metrics = {'mae': float(overall_mae), 'rmse': float(overall_rmse)}
        self.logger.info(f"Detailed testing completed. Overall MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}")
        return all_window_results, overall_metrics 