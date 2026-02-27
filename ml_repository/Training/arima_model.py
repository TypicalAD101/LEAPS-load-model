import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from Support.config_handler import ConfigHandler
from utils.logger import get_logger

class arima_model:
    """
    Trainer for ARIMA model for time series forecasting.
    """
    def __init__(self):
        self.logger = get_logger()
        self.config_handler = ConfigHandler()
        self.data_path = self.config_handler.get_data_path()

    def train(self, order=(1,1,1), model_save_path: str = None, **kwargs):
        """
        Train an ARIMA model and save it to the specified path.
        Args:
            order: ARIMA order (p,d,q)
            model_save_path: Path to save the trained model
        """
        self.logger.info('Starting ARIMA model training.')
        df = pd.read_csv(self.data_path)
        y = df['criticalLoadForecast_0'].values.astype(float)
        self.logger.info(f'Fitting ARIMA model with order={order} on {len(y)} samples.')
        model = ARIMA(y, order=order)
        model_fit = model.fit()
        if model_save_path is None:
            model_save_path = 'model_arima.pkl'
        joblib.dump(model_fit, model_save_path)
        self.logger.info(f'ARIMA model saved to {model_save_path}')
        return model_fit 