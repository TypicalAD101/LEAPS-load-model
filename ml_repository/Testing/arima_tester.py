import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Support.config_handler import ConfigHandler
from utils.logger import get_logger

class ArimaTester:
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.config_handler = ConfigHandler()
        self.data_path = self.config_handler.get_data_path()

    def test(self, model_path, seq_length=96):
        self.logger.info(f'Testing ARIMA model from {model_path}')
        df = pd.read_csv(self.data_path)
        y = df['criticalLoadForecast_0'].values.astype(float)
        n_total = len(y)
        test_start = int(n_total * 0.9)
        y_test = y[test_start:]
        model_fit = joblib.load(model_path)
        # Forecast the test period
        forecast = model_fit.predict(start=test_start, end=n_total-1)
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        r2 = r2_score(y_test, forecast)
        mape = np.mean(np.abs((y_test - forecast) / np.where(y_test != 0, y_test, 1))) * 100
        print("\n" + "="*60)
        print("ARIMA TEST RESULTS")
        print("="*60)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print("Sample predictions (first 10):")
        for i in range(min(10, len(forecast))):
            print(f"Actual: {y_test[i]:.4f}\tPredicted: {forecast[i]:.4f}\tDiff: {abs(y_test[i]-forecast[i]):.4f}")
        print("="*60)
        self.logger.info('ARIMA testing completed.')
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape} 