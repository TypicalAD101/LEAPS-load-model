import logging
from Validation.base_validator import BaseValidator

class DartsValidator(BaseValidator):
    """
    Validator for Darts models.
    Inherits from BaseValidator and implements model-specific logic for Darts.
    """
    def run_validation(self, model_path, test_data, actuals, seq_length=96, print_results=False):
        """
        Validate the Darts model by predicting full 96-step chunks, not just one step ahead.
        """
        import numpy as np
        from darts import TimeSeries
        self.logger.info(f'Starting validation for model at {model_path}')
        model = self._load_model(model_path)

        # Ensure test_data is a Darts TimeSeries
        if not isinstance(test_data, TimeSeries):
            test_data = TimeSeries.from_values(test_data)

        predictions = []
        actual_values = []
        abs_diffs = []
        squared_diffs = []

        # Validate on non-overlapping chunks of the test data
        for i in range(0, len(test_data) - 2 * seq_length + 1, seq_length):
            # 1. Define the historical data chunk for input
            history_chunk = test_data[i : i + seq_length]
            # 2. Define the ground truth for the forecast period
            actual_chunk = test_data[i + seq_length : i + 2 * seq_length]
            # Ensure we have a full chunk to compare against
            if len(actual_chunk) < seq_length:
                continue
            # 3. Predict the entire next chunk (96 steps)
            predicted_chunk = model.predict(n=seq_length, series=history_chunk)
            pred_vals = predicted_chunk.values().flatten()
            act_vals = actual_chunk.values().flatten()
            predictions.extend(pred_vals)
            actual_values.extend(act_vals)
            abs_diffs.extend(np.abs(act_vals - pred_vals))
            squared_diffs.extend((act_vals - pred_vals) ** 2)
            if print_results:
                for j in range(seq_length):
                    print(f"Chunk {i//seq_length}, Step {j}: Actual: {act_vals[j]:.6f}, Predicted: {pred_vals[j]:.6f}, Abs Diff: {abs(act_vals[j] - pred_vals[j]):.6f}")

        # Calculate and display summary statistics
        self._display_summary_statistics(actual_values, predictions, abs_diffs, squared_diffs, seq_length=seq_length)
        self.logger.info(f"Darts validation complete. {len(predictions)} predictions.")

    def _load_model(self, model_path):
        from darts.models import NBEATSModel  # or the correct Darts model class
        self.logger.info(f"Loading Darts model from {model_path}")
        return NBEATSModel.load(model_path)

    def _predict_one_step(self, model, input_data):
        # No longer used; present to satisfy abstract base class
        raise NotImplementedError("_predict_one_step is not used in DartsValidator.") 