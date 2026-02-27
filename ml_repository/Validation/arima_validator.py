import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Support.config_handler import ConfigHandler
from utils.logger import get_logger

class ArimaValidator:
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.config_handler = ConfigHandler()
        self.data_path = self.config_handler.get_data_path()

    def run_validation(self, model_path, seq_length=96):
        self.logger.info(f'Validating ARIMA model from {model_path}')
        df = pd.read_csv(self.data_path)
        y = df['criticalLoadForecast_0'].values.astype(float)
        n_total = len(y)
        train_end = int(n_total * 0.8)
        val_end = int(n_total * 0.9)
        y_val = y[train_end:val_end]
        model_fit = joblib.load(model_path)
        # Forecast the validation period
        forecast = model_fit.predict(start=train_end, end=val_end-1)
        mae = mean_absolute_error(y_val, forecast)
        rmse = np.sqrt(mean_squared_error(y_val, forecast))
        r2 = r2_score(y_val, forecast)
        mape = np.mean(np.abs((y_val - forecast) / np.where(y_val != 0, y_val, 1))) * 100
        print("\n" + "="*60)
        print("ARIMA VALIDATION RESULTS")
        print("="*60)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print("Sample predictions (first 10):")
        for i in range(min(10, len(forecast))):
            print(f"Actual: {y_val[i]:.4f}\tPredicted: {forecast[i]:.4f}\tDiff: {abs(y_val[i]-forecast[i]):.4f}")
        print("="*60)

        # Per-day (96-chunk) metrics
        num_days = len(y_val) // seq_length
        if num_days > 0:
            daily_mae = []
            daily_rmse = []
            daily_accuracy = []
            for day in range(num_days):
                start = day * seq_length
                end = start + seq_length
                day_preds = forecast[start:end]
                day_actuals = y_val[start:end]
                day_mae = np.mean(np.abs(day_preds - day_actuals))
                day_rmse = np.sqrt(np.mean((day_preds - day_actuals) ** 2))
                mean_actual = np.mean(np.abs(day_actuals))
                # Define accuracy as 1 - (MAE/mean_actual), clip to [0, 1]
                day_acc = 1 - (day_mae / mean_actual) if mean_actual != 0 else 0.0
                day_acc = max(0.0, min(1.0, day_acc))
                daily_mae.append(day_mae)
                daily_rmse.append(day_rmse)
                daily_accuracy.append(day_acc)
            print("\nPer-Day (96-chunk) Metrics:")
            print(f"Number of days: {num_days}")
            print(f"Mean Daily MAE: {np.mean(daily_mae):.6f}")
            print(f"Mean Daily RMSE: {np.mean(daily_rmse):.6f}")
            print(f"Median Daily MAE: {np.median(daily_mae):.6f}")
            print(f"Median Daily RMSE: {np.median(daily_rmse):.6f}")
            print("Sample Daily MAE (first 5 days):", daily_mae[:5])
            print("Sample Daily RMSE (first 5 days):", daily_rmse[:5])
            print(f"\nMean Daily Accuracy: {np.mean(daily_accuracy):.4f}")
            print(f"Median Daily Accuracy: {np.median(daily_accuracy):.4f}")
            print("Sample Daily Accuracy (first 5 days):", daily_accuracy[:5])
        else:
            print("\nNot enough data for per-day (96-chunk) metrics.")

        self.logger.info('ARIMA validation completed.')
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape} 