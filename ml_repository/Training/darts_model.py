from Training.base_trainer import BaseTrainer
import torch
from darts.models import NBEATSModel
from utils.logger import get_logger

class darts_model(BaseTrainer):
    """
    Trainer for Darts NBEATSModel for time series forecasting.
    """
    def __init__(self):
        self.model_name = None
        self.model_path = None
        self.logger = get_logger()

    def run(self):
        """
        Placeholder for run logic.
        """
        self.logger.debug('Run method called (not implemented).')

    def prepare_data(self):
        """
        Placeholder for data preparation logic.
        """
        self.logger.debug('Prepare data method called (not implemented).')

    def train(self, train_series, val_series, model_name=None, model_save_path: str = None, epochs=5, **kwargs):
        """
        Train a Darts NBEATSModel and save it to the specified path.
        Args:
            train_series: Training time series.
            val_series: Validation time series.
            model_name: Optional model name.
            model_save_path: Path to save the trained model.
            epochs: Number of training epochs (from config).
        """
        self.logger.info('Starting Darts model training.')
        self.logger.info(f'Training parameters: epochs={epochs}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f'Using device: {device}')

        model = NBEATSModel(
            input_chunk_length=24 * 4,
            output_chunk_length=96,
            batch_size=128,
            n_epochs=epochs,
            dropout=0.1,
            activation="ReLU",
            random_state=42,
        )
        self.logger.debug('NBEATSModel instantiated.')

        model.fit(
            series=train_series,
            val_series=val_series,
            verbose=True
        )
        self.logger.debug('Model training completed.')
        if model_save_path is None:
            model_save_path = 'model_darts.h5'
        model.save(model_save_path)
        self.logger.info(f'Model saved to {model_save_path}')

    def validation(self):
        """
        Placeholder for validation logic.
        """
        self.logger.debug('Validation method called (not implemented).') 