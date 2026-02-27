import logging
from abc import ABC, abstractmethod

class BaseValidator(ABC):
    """
    BaseValidator provides a generic validation pipeline for model validators.
    Handles model loading, iteration, prediction, and result display.
    Child classes must implement model-specific logic for loading and predicting.
    """
    def __init__(self, logger=None):
        """
        Initialize the BaseValidator.
        Args:
            logger (logging.Logger, optional): Logger instance. If None, uses root logger.
        """
        self.logger = logger or logging.getLogger(__name__)

    def run_validation(self, model_path, test_data, actuals, seq_length=96, print_results=False):
        """
        Run the validation loop: load model, iterate over data, predict, and display results.
        Args:
            model_path (str): Path to the saved model.
            test_data (array-like): Test input data (e.g., X_test or val_series).
            actuals (array-like): Ground truth values (e.g., y_test or val_series).
            seq_length (int): Sequence length for sliding window.
            print_results (bool): If True, print each validation result to the console.
        """
        import numpy as np
        self.logger.info(f'Starting validation for model at {model_path}')
        model = self._load_model(model_path)

        def to_scalar(val):
            if isinstance(val, (np.ndarray, list)):
                val = val[0] if len(val) > 0 else 0.0
            if hasattr(val, 'item'):
                val = val.item()
            return float(val)

        # Lists to store all results for summary statistics
        all_actuals = []
        all_predictions = []
        all_abs_diffs = []
        all_squared_diffs = []

        for i in range(len(test_data) - seq_length):
            X_win = test_data[i]
            X_window = self._prepare_input(X_win, seq_length)
            predicted = self._predict_one_step(model, X_window)
            actual = self._get_actual_value(actuals, i, seq_length)
            abs_diff = abs(actual - predicted)
            
            # Convert all to Python float scalars
            actual_scalar = to_scalar(actual)
            predicted_scalar = to_scalar(predicted)
            abs_diff_scalar = to_scalar(abs_diff)
            
            # Store for summary statistics
            all_actuals.append(actual_scalar)
            all_predictions.append(predicted_scalar)
            all_abs_diffs.append(abs_diff_scalar)
            all_squared_diffs.append(abs_diff_scalar ** 2)
            
            if print_results:
                print(f"Index: {i}, Actual: {actual_scalar:.6f}, Predicted: {predicted_scalar:.6f}, Abs Diff: {abs_diff_scalar:.6f}")

        # Calculate and display summary statistics (now with per-day metrics)
        self._display_summary_statistics(all_actuals, all_predictions, all_abs_diffs, all_squared_diffs, seq_length=seq_length)
        
        self.logger.info(f"Validation completed for {len(all_actuals)} predictions")

    def _display_summary_statistics(self, actuals, predictions, abs_diffs, squared_diffs, seq_length=96):
        """
        Calculate and display summary statistics for validation results, including per-day (96-chunk) metrics.
        """
        import numpy as np
        
        # Convert to numpy arrays for calculations
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        abs_diffs = np.array(abs_diffs)
        squared_diffs = np.array(squared_diffs)
        
        # Calculate global metrics
        mae = np.mean(abs_diffs)
        rmse = np.sqrt(np.mean(squared_diffs))
        mse = np.mean(squared_diffs)
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        min_error = np.min(abs_diffs)
        max_error = np.max(abs_diffs)
        std_error = np.std(abs_diffs)
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY STATISTICS")
        print("="*60)
        print(f"Total Predictions: {len(actuals)}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
        print(f"Mean Square Error (MSE): {mse:.6f}")
        print(f"R-squared (R²): {r_squared:.6f}")
        print(f"Minimum Absolute Error: {min_error:.6f}")
        print(f"Maximum Absolute Error: {max_error:.6f}")
        print(f"Standard Deviation of Errors: {std_error:.6f}")
        print("="*60)

        # Per-day (96-chunk) metrics
        num_days = len(actuals) // seq_length
        if num_days > 0:
            daily_mae = []
            daily_rmse = []
            for day in range(num_days):
                start = day * seq_length
                end = start + seq_length
                day_preds = predictions[start:end]
                day_actuals = actuals[start:end]
                daily_mae.append(np.mean(np.abs(day_preds - day_actuals)))
                daily_rmse.append(np.sqrt(np.mean((day_preds - day_actuals) ** 2)))
            print("\nPer-Day (96-chunk) Metrics:")
            print(f"Number of days: {num_days}")
            print(f"Mean Daily MAE: {np.mean(daily_mae):.6f}")
            print(f"Mean Daily RMSE: {np.mean(daily_rmse):.6f}")
            print(f"Median Daily MAE: {np.median(daily_mae):.6f}")
            print(f"Median Daily RMSE: {np.median(daily_rmse):.6f}")
            print("Sample Daily MAE (first 5 days):", daily_mae[:5])
            print("Sample Daily RMSE (first 5 days):", daily_rmse[:5])
        else:
            print("\nNot enough data for per-day (96-chunk) metrics.")

    @abstractmethod
    def _load_model(self, model_path):
        """
        Load and return the model from the given path.
        Args:
            model_path (str): Path to the saved model.
        Returns:
            Loaded model object.
        """
        pass

    @abstractmethod
    def _predict_one_step(self, model, input_data):
        """
        Make a prediction for a single input window.
        Args:
            model: Loaded model object.
            input_data: Prepared input data for prediction.
        Returns:
            Predicted value (float).
        """
        pass

    def _prepare_input(self, X_win, seq_length):
        """
        Prepare the input window for prediction. Can be overridden if needed.
        Args:
            X_win: Raw input window.
            seq_length (int): Sequence length.
        Returns:
            Prepared input data.
        """
        import numpy as np
        X_window = np.array(X_win).flatten().reshape(seq_length, 1)
        return X_window

    def _get_actual_value(self, actuals, i, seq_length):
        """
        Get the actual value for the current window. Can be overridden for custom indexing.
        Args:
            actuals: Array of ground truth values.
            i (int): Current index in the loop.
            seq_length (int): Sequence length.
        Returns:
            Actual value (float).
        """
        # Default: use actuals[i + seq_length] (can be overridden)
        try:
            return actuals[i + seq_length]
        except IndexError:
            return actuals[i]  # fallback 