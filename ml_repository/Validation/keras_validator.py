import logging
from Validation.base_validator import BaseValidator
import tensorflow as tf

class KerasValidator(BaseValidator):
    """
    Validator for standard Keras models.
    Inherits from BaseValidator and implements model-specific logic for Keras.
    """
    def run_validation(self, model_path, test_data, actuals, seq_length=96, scaler=None, print_results=False):
        import numpy as np
        self.logger.info(f'Starting recursive multi-step validation for model at {model_path}')
        model = self._load_model(model_path)
        if scaler is None:
            self.logger.warning('Scaler not provided; results will be in normalized space!')
        X, y = test_data  # X: (num_samples, seq_length, 1), y: (num_samples, 1)
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
                true_future.append(y[start_idx + j][0])
            # Recursive forecast
            preds = []
            current_window = history.copy()
            for step in range(seq_length):
                pred = model.predict(current_window, verbose=0)
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
        # Calculate metrics
        self._display_summary_statistics(all_actuals, all_preds, np.abs(np.array(all_actuals) - np.array(all_preds)), (np.array(all_actuals) - np.array(all_preds))**2, seq_length=seq_length)
        self.logger.info(f"Recursive validation completed for {len(all_preds)} predictions.")

    def _load_model(self, model_path):
        self.logger.info(f"Loading Keras model from {model_path}")
        return tf.keras.models.load_model(model_path)

    def _predict_one_step(self, model, input_data):
        pred = model.predict(input_data)
        return pred[0][0]

    def _get_actual_value(self, actuals, i, seq_length):
        # For Keras, actuals are indexed by i
        try:
            return actuals[i]
        except IndexError:
            return actuals[-1] 