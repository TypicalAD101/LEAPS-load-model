import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Support.config_handler import ConfigHandler


class DataHandler:
    def __init__(self, config_handler=None):
        self.raw_data = None
        self.config_handler = config_handler or ConfigHandler()
        self.data_path = self.config_handler.get_data_path()
        # self.run()

    def run(self):
        self.data_loading()
        # self.data_visualization()

    def data_loading(self):
        df = pd.read_csv(self.data_path)
        self.raw_data = df.iloc[:, 1].values

    # def data_visualization(self):
    #     plt.plot(self.raw_data)
    #     plt.title("Plot of Data")
    #     plt.xlabel("Index")
    #     plt.ylabel("Value")
    #     plt.show()

    def preprocess_data(self, data_type='train', seq_length=96):
        """
        Unified preprocessing for time series data.
        If the model is Darts, always use preprocess_data_darts (returns TimeSeries).
        Otherwise, use file-based logic.
        """
        training_type = self.config_handler.get_training_type()
        if training_type == 'd':
            return self.preprocess_data_darts(data_type)
        import os
        if os.path.basename(self.data_path) == 'Load_Profile.csv':
            return self.preprocess_data_loadprofile(data_type, seq_length)
        # Otherwise, fallback to old logic
        if training_type == 'k':
            return self.preprocess_data_keras(data_type, seq_length)
        elif training_type == 't':
            return self.preprocess_data_keras_tuner(data_type, seq_length)
        else:
            raise ValueError(f"Unsupported training type: {training_type}")

    def preprocess_data_keras(self, data_type='train', seq_length=None):
        """
        Preprocess for Keras: windowing, scaling, day-aligned chronological split.
        Returns (X, y) for the requested split.
        """
        df = pd.read_csv(self.data_path)
        values = df['crit_load'].values  # Use consistent column name
        if seq_length is None:
            seq_length = self.config_handler.get_model_config()['sequence_length']
        steps_per_day = seq_length
        n_total = len(values)
        # Train-Max split logic
        num_test_windows = self.config_handler.get_training_config().get('num_test_windows', 3)
        test_set_size = (num_test_windows * steps_per_day) + steps_per_day
        if data_type == 'train_max':
            test_data = values[-test_set_size:]
            train_data = values[:-test_set_size]
            # Fit scaler only on training data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
            test_scaled = self.scaler.transform(test_data.reshape(-1, 1)).flatten()
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)
            X_train, y_train = create_sequences(train_scaled, seq_length)
            X_test, y_test = create_sequences(test_scaled, seq_length)
            return (X_train.reshape(-1, seq_length, 1), y_train.reshape(-1, 1)), (X_test.reshape(-1, seq_length, 1), y_test.reshape(-1, 1))
        # Day-aligned split indices
        train_end = (int(n_total * 0.8) // steps_per_day) * steps_per_day
        val_end = (int(n_total * 0.9) // steps_per_day) * steps_per_day
        # Split data first, then scale
        train_values = values[:train_end]
        val_values = values[train_end:val_end]
        test_values = values[val_end:]
        # Fit scaler only on training data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
        val_scaled = self.scaler.transform(val_values.reshape(-1, 1)).flatten()
        test_scaled = self.scaler.transform(test_values.reshape(-1, 1)).flatten()
        # Create sequences for each split
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        if data_type == 'train':
            X, y = create_sequences(train_scaled, seq_length)
            return X.reshape(-1, seq_length, 1), y.reshape(-1, 1)
        elif data_type == 'validation':
            X, y = create_sequences(val_scaled, seq_length)
            return X.reshape(-1, seq_length, 1), y.reshape(-1, 1)
        elif data_type == 'test':
            X, y = create_sequences(test_scaled, seq_length)
            return X.reshape(-1, seq_length, 1), y.reshape(-1, 1)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    def preprocess_data_keras_tuner(self, data_type='train', seq_length=None):
        """
        Preprocess for Keras Tuner: windowing, scaling, day-aligned chronological split.
        Returns (X, y) for the requested split.
        """
        df = pd.read_csv(self.data_path)
        values = df['crit_load'].values  # assumes 'crit_load' is the target
        if seq_length is None:
            seq_length = self.config_handler.get_model_config()['sequence_length']
        steps_per_day = seq_length
        n_total = len(values)
        train_end = (int(n_total * 0.8) // steps_per_day) * steps_per_day
        val_end = (int(n_total * 0.9) // steps_per_day) * steps_per_day
        # Fit scaler only on train
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = self.scaler.fit_transform(values[:train_end].reshape(-1, 1))
        scaled_all = np.concatenate([
            scaled_train,
            self.scaler.transform(values[train_end:].reshape(-1, 1))
        ])
        # Windowing
        X, y = [], []
        for i in range(n_total - seq_length):
            X.append(scaled_all[i:i + seq_length].flatten())
            y.append(scaled_all[i + seq_length][0])
        X = np.array(X)
        y = np.array(y)
        # Day-aligned split
        n_samples = len(X)
        train_idx = (int(n_samples * 0.8) // steps_per_day) * steps_per_day
        val_idx = (int(n_samples * 0.9) // steps_per_day) * steps_per_day
        if data_type == 'train':
            return X[:train_idx], y[:train_idx]
        elif data_type == 'validation':
            return X[train_idx:val_idx], y[train_idx:val_idx]
        elif data_type == 'test':
            return X[val_idx:], y[val_idx:]
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    def preprocess_data_darts(self, data_type='train'):
        """
        Preprocess for Darts: TimeSeries object, scaling, chronological split.
        Returns TimeSeries for the requested split.
        Supports both 'crit_load' and 'criticalLoadForecast_0' columns.
        """
        import pandas as pd
        from darts import TimeSeries
        from darts.dataprocessing.transformers import Scaler
        df = pd.read_csv(self.data_path)
        # Support both column names
        if 'crit_load' in df.columns:
            values = df['crit_load'].values
        elif 'criticalLoadForecast_0' in df.columns:
            values = df['criticalLoadForecast_0'].values.astype(float)
        else:
            raise ValueError("No valid target column found in data file.")
        n_total = len(values)
        train_end = int(n_total * 0.8)
        val_end = int(n_total * 0.9)
        # Create TimeSeries
        series = TimeSeries.from_values(values)
        scaler = Scaler()
        train_series = series[:train_end]
        scaler.fit(train_series)
        scaled_series = scaler.transform(series)
        if data_type == 'train':
            return scaled_series[:train_end]
        elif data_type == 'validation':
            return scaled_series[train_end:val_end]
        elif data_type == 'test':
            return scaled_series[val_end:]
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    def preprocess_data_loadprofile(self, data_type='train', seq_length=None):
        """
        Preprocess the new LoadProfile.csv file for ML training.
        - Uses criticalLoadForecast_0 as the main data
        - Returns windowed, scaled data for the requested split
        - Day-aligned splits
        """
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        df = pd.read_csv(self.data_path)
        values = df['criticalLoadForecast_0'].values.astype(float)
        if seq_length is None:
            seq_length = self.config_handler.get_model_config()['sequence_length']
        steps_per_day = seq_length
        n_total = len(values)
        num_test_windows = self.config_handler.get_training_config().get('num_test_windows', 3)
        test_set_size = (num_test_windows * steps_per_day) + steps_per_day
        if data_type == 'train_max':
            test_data = values[-test_set_size:]
            train_data = values[:-test_set_size]
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
            test_scaled = self.scaler.transform(test_data.reshape(-1, 1)).flatten()
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)
            X_train, y_train = create_sequences(train_scaled, seq_length)
            X_test, y_test = create_sequences(test_scaled, seq_length)
            return (X_train.reshape(-1, seq_length, 1), y_train.reshape(-1, 1)), (X_test.reshape(-1, seq_length, 1), y_test.reshape(-1, 1))
        # Day-aligned split indices
        train_end = (int(n_total * 0.8) // steps_per_day) * steps_per_day
        val_end = (int(n_total * 0.9) // steps_per_day) * steps_per_day
        # Split data first, then scale
        train_values = values[:train_end]
        val_values = values[train_end:val_end]
        test_values = values[val_end:]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
        val_scaled = self.scaler.transform(val_values.reshape(-1, 1)).flatten()
        test_scaled = self.scaler.transform(test_values.reshape(-1, 1)).flatten()
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        if data_type == 'train':
            X, y = create_sequences(train_scaled, seq_length)
            return X.reshape(-1, seq_length, 1), y.reshape(-1, 1)
        elif data_type == 'validation':
            X, y = create_sequences(val_scaled, seq_length)
            return X.reshape(-1, seq_length, 1), y.reshape(-1, 1)
        elif data_type == 'test':
            X, y = create_sequences(test_scaled, seq_length)
            return X.reshape(-1, seq_length, 1), y.reshape(-1, 1)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    def postprocess_data(self):
        pass

    def gif_visualization(self):
        pass
