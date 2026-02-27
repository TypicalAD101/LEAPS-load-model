import configparser
from pathlib import Path
import os
from utils.model_enums import ModelType, ConfigKeys
from utils.config_enums import ModelPaths, DatabaseNames, TrainingParams

class ConfigHandler:
    def __init__(self, config_path: str = None):
        self.config = configparser.ConfigParser()
        if config_path is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = os.path.join(script_dir, 'config.ini')
        else:
            self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        self.config.read(self.config_path)

    def get_model_config(self) -> dict:
        """Get model-related configuration"""
        return {
            'train_mode': self.config['Model'][ConfigKeys.TRAIN_MODE.value],
            'val_mode': self.config['Model'][ConfigKeys.VAL_MODE.value],
            'impl_mode': self.config['Model'][ConfigKeys.IMPL_MODE.value],
            'sequence_length': self.config.getint('Model', ConfigKeys.SEQUENCE_LENGTH.value)
        }

    def get_paths(self) -> dict:
        """Get model paths"""
        return {
            ModelType.KERAS.value: self.config['Paths'][ModelPaths.KERAS.value],
            ModelType.KERAS_TUNER.value: self.config['Paths'][ModelPaths.KERAS_TUNER.value],
            ModelType.DARTS.value: self.config['Paths'][ModelPaths.DARTS.value],
            ModelType.ARIMA.value: self.config['Paths'].get('model_arima', 'model_arima.pkl')
        }

    def get_database_config(self) -> dict:
        """Get database configuration"""
        return {
            ModelType.KERAS.value: self.config['Database'][DatabaseNames.KERAS.value],
            ModelType.KERAS_TUNER.value: self.config['Database'][DatabaseNames.KERAS_TUNER.value],
            ModelType.DARTS.value: self.config['Database'][DatabaseNames.DARTS.value],
            ModelType.ARIMA.value: self.config['Database'].get('arima_val_db', 'arima_val.db')
        }

    def get_training_config(self) -> dict:
        """Get training configuration"""
        return {
            'batch_size': self.config.getint('Training', TrainingParams.BATCH_SIZE.value),
            'epochs': self.config.getint('Training', TrainingParams.EPOCHS.value),
            'validation_split': self.config.getfloat('Training', TrainingParams.VALIDATION_SPLIT.value)
        }

    def get_data_path(self) -> str:
        """Get the data file path from config"""
        return self.config['Paths']['data_path']
    
    def get_config(self) -> dict:
        """Get the complete configuration as a dictionary"""
        config_dict = {}
        
        # Add all sections
        for section in self.config.sections():
            config_dict[section] = {}
            for key, value in self.config[section].items():
                # Try to convert to int or float if possible
                try:
                    if '.' in value:
                        config_dict[section][key] = float(value)
                    else:
                        config_dict[section][key] = int(value)
                except ValueError:
                    config_dict[section][key] = value
        
        return config_dict
    
    def get_training_type(self) -> str:
        """Get the current training type"""
        return self.config['Model'][ConfigKeys.TRAIN_MODE.value] 