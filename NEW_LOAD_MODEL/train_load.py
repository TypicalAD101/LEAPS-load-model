# train_load.py — NEW LOAD MODEL TRAINING SCRIPT
# Forked from trainCPU (1).py
# KEY CHANGES:
#   1. REMOVED daily seasonal differencing (lag=96)
#   2. ADDED 7 cyclical time features from timestamps → input shape (96, 8)
#   3. MinMaxScaler fitted on RAW load values (not differenced)
#   4. Output remains (96, 1) — only load is predicted
#
# LSTM Time Series Forecasting Model - DIRECT MANY-TO-MANY VERSION
# - Many-to-many architecture: Input 96 steps → Output 96 steps in ONE forward pass
# - NO recursion, NO error accumulation
# - NO differencing — model predicts absolute load values directly
# - 7 cyclical time features give the model awareness of hour, day-of-week, month
# - MinMaxScaler normalization with production export
# - Multi-GPU training with mixed precision

import os, sys
if "--cpu_only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import time
import warnings
import logging
import keras_tuner
import gc
import psutil
import subprocess

# =============================================================================
# 1. LOGGING AND GPU CONFIGURATION
# =============================================================================
def setup_logging(log_dir):
    """Sets up logging to file and console."""
    log_filename = os.path.join(log_dir, 'training_log.log')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured.")

def log_memory_usage(stage=""):
    """Logs current memory usage for monitoring."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        logging.info(f"Memory Usage {stage}: {memory_info.rss / 1024**3:.2f} GB "
                    f"({memory_percent:.1f}% of system)")
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, gpu in enumerate(gpus):
                    gpu_memory = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    if gpu_memory:
                        logging.info(f"GPU {i} Memory: {gpu_memory['current'] / 1024**3:.2f} GB")
        except:
            pass
    except Exception as e:
        logging.warning(f"Could not log memory usage: {e}")

def configure_gpu():
    """Configure GPU for optimal HPC performance with multi-GPU support."""
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices:
        logging.info(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices}")
    tf_force_gpu_growth = os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'false').lower() == 'true'
    if tf_force_gpu_growth:
        logging.info("TF_FORCE_GPU_ALLOW_GROWTH is enabled")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                if not tf_force_gpu_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Memory growth enabled for GPU {gpu}")
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=75000)]
                    )
                    logging.info(f"Memory limit set to 75GB for GPU {gpu}")
                except Exception as e:
                    logging.warning(f"Could not set memory limit for GPU {gpu}: {e}")
            logging.info(f"GPU configured: {len(physical_devices)} GPU(s) available.")
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logging.info(f"Mixed precision enabled: {policy.name}")
            except Exception as e:
                logging.warning(f"Mixed precision not available: {e}")
                policy = tf.keras.mixed_precision.Policy('float32')
                tf.keras.mixed_precision.set_global_policy(policy)
                logging.info(f"Using float32 precision: {policy.name}")
        except RuntimeError as e:
            logging.error(f"GPU configuration error: {e}")
    else:
        logging.info("No GPU available, using CPU.")
    try:
        tf.config.optimizer.set_jit(True)
        logging.info("XLA compilation enabled for optimal performance.")
    except Exception as e:
        logging.warning(f"XLA compilation not available: {e}")
    if os.environ.get('TF_DEBUG_DEVICE_PLACEMENT', 'false').lower() == 'true':
        tf.debugging.set_log_device_placement(True)
        logging.info("Device placement logging enabled for multi-GPU debugging")
    else:
        logging.info("Device placement logging disabled (set TF_DEBUG_DEVICE_PLACEMENT=true to enable)")
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "pin_to_host_optimization": False,
    })
    logging.info("Advanced TensorFlow optimizations enabled for HPC performance")
    tf_vars = ['TF_GPU_THREAD_MODE', 'TF_GPU_THREAD_COUNT', 'TF_USE_CUDNN']
    logging.info("TensorFlow Environment Variables:")
    for var in tf_vars:
        value = os.environ.get(var, 'Not Set')
        logging.info(f"  {var}: {value}")


@tf.keras.utils.register_keras_serializable(package="Custom")
class ClipByValue(tf.keras.layers.Layer):
    def __init__(self, min_value, max_value, **kwargs):
        super().__init__(**kwargs)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def call(self, x):
        return tf.clip_by_value(x, self.min_value, self.max_value)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"min_value": self.min_value, "max_value": self.max_value})
        return cfg


# =============================================================================
# 2. TIME FEATURE ENGINEERING (NEW)
# =============================================================================
def compute_time_features(timestamps):
    """
    Compute 7 cyclical time features from an array of datetime timestamps.
    
    Features:
        0: hour_sin   = sin(2π * hour / 24)
        1: hour_cos   = cos(2π * hour / 24)
        2: dow_sin    = sin(2π * day_of_week / 7)
        3: dow_cos    = cos(2π * day_of_week / 7)
        4: month_sin  = sin(2π * (month-1) / 12)
        5: month_cos  = cos(2π * (month-1) / 12)
        6: is_weekend = 1.0 if Saturday/Sunday, else 0.0
    
    Args:
        timestamps: array-like of pd.Timestamp / datetime objects
    
    Returns:
        np.ndarray of shape (len(timestamps), 7), dtype float32
    """
    ts = pd.DatetimeIndex(timestamps)
    hour_frac = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek  # Monday=0, Sunday=6
    month = ts.month    # 1-12

    features = np.column_stack([
        np.sin(2 * np.pi * hour_frac / 24.0),
        np.cos(2 * np.pi * hour_frac / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * (month - 1) / 12.0),
        np.cos(2 * np.pi * (month - 1) / 12.0),
        (dow >= 5).astype(np.float32),
    ]).astype(np.float32)

    return features


# =============================================================================
# 3. ADVANCED v3 DATA PROCESSING CLASS (MODIFIED — NO DIFFERENCING)
# =============================================================================
class TimeSeriesDataProcessor:
    """
    Handles data loading, cleaning, and preparation for the model using
    sophisticated v3 preprocessing with gap classification and hierarchical filling.
    
    KEY CHANGE: No differencing. Returns both values and timestamps.
    """
    def __init__(self, seq_length=96, forecast_horizon=96, batch_size=1024,
                 dataset_kind="electric", n_input_features=8):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.dataset_kind = dataset_kind
        self.n_input_features = n_input_features  # 1 (load) + 7 (time features) = 8

    def classify_gaps(self, s, short_hours=7, points_per_hour=4):
        """Return boolean masks for short vs long gaps based on consecutive NaNs."""
        na = s.isna()
        run_id = (na != na.shift()).cumsum()
        run_len = na.groupby(run_id).transform('sum')
        short_limit = short_hours * points_per_hour
        short_mask = na & (run_len < short_limit)
        long_mask = na & (run_len >= short_limit)
        return short_mask, long_mask

    def preprocess_v3_frame(self, df, short_hours=7):
        """Apply v3 preprocessing to a single time-indexed frame with column 'value'.
        Returns DataFrame with 'value_filled' column and datetime index preserved."""
        cols = [c.strip().lower() for c in df.columns]
        time_col = df.columns[cols.index('time')] if 'time' in cols else None
        val_col = df.columns[cols.index('value')] if 'value' in cols else None
        if time_col is None or val_col is None:
            raise ValueError("Frame must contain 'time' and 'value' columns.")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

        s = pd.to_numeric(df[val_col], errors='coerce')
        if self.dataset_kind != "solar":
            s = s.replace(0, np.nan)

        total_points = len(s)
        zero_points = (s == 0).sum()
        nan_points = s.isna().sum()
        logging.info(f"Initial data: {total_points} points, {zero_points} zeros, {nan_points} NaN values")

        points_per_hour = 4
        short_mask, long_mask = self.classify_gaps(s, short_hours=short_hours, points_per_hour=points_per_hour)
        short_gaps = short_mask.sum()
        long_gaps = long_mask.sum()
        logging.info(f"Gap classification: {short_gaps} short gaps (<{short_hours}h), {long_gaps} long gaps (≥{short_hours}h)")

        lags = [96, 2*96, 3*96, 4*96, 5*96, 6*96, 7*96]
        cand = pd.concat([s.shift(L) for L in lags], axis=1)
        cand.columns = [f"lag_{L}" for L in lags]
        prev_days_first = cand.bfill(axis=1).iloc[:, 0]
        tod_key = pd.Series(s.index.strftime('%H:%M'), index=s.index)
        tod_medians = s.groupby(tod_key).median()
        tod_fill = tod_key.map(tod_medians)
        locf = s.ffill()

        short_fill = prev_days_first.copy()
        short_fill = short_fill.where(short_fill.notna(), tod_fill)
        short_fill = short_fill.where(short_fill.notna(), locf)
        after_short = s.copy()
        after_short.loc[short_mask] = short_fill.loc[short_mask]

        filled = after_short.copy()
        filled.loc[long_mask] = tod_fill.loc[long_mask]

        remaining_nans = filled.isna().sum()
        logging.info(f"After v3 preprocessing: {remaining_nans} NaN values remain out of {total_points} total points")

        out = pd.DataFrame(index=df.index)
        out['value_filled'] = filled
        return out

    def load_and_process_data(self, data_dir):
        """Loads all CSVs from a directory, applies v3 preprocessing, and combines them.
        Returns DataFrame with datetime index and 'value_filled' column."""
        logging.info(f"Loading and applying v3 preprocessing to all data from directory: {data_dir}")
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        logging.info(f"Found {len(csv_files)} CSV files to process")
        if not csv_files:
            raise RuntimeError("No CSV files found in data_dir.")

        batch_size = 5
        all_dfs = []
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            batch_dfs = []
            for filename in batch_files:
                file_path = os.path.join(data_dir, filename)
                logging.info(f"Processing {filename} with v3 preprocessing...")
                try:
                    df = pd.read_csv(file_path)
                    _cols = {c.strip().lower(): c for c in df.columns}
                    _rename = {}
                    if 'time' in _cols:
                        _rename[_cols['time']] = 'time'
                    else:
                        for _k in ('localtime', 'local_time', 'datetime', 'date_time', 'timestamp', 'date', 'dt'):
                            if _k in _cols:
                                _rename[_cols[_k]] = 'time'
                                break
                    if 'value' in _cols:
                        _rename[_cols['value']] = 'value'
                    else:
                        for _k in ('power(mw)', 'power_mw', 'power', 'mw', 'generation', 'gen', 'p'):
                            if _k in _cols:
                                _rename[_cols[_k]] = 'value'
                                break
                    if _rename:
                        df = df.rename(columns=_rename)
                    if 'time' not in df.columns or 'value' not in df.columns:
                        raise ValueError(f"Could not map columns to time/value. Columns found: {list(df.columns)}")
                    df['value'] = pd.to_numeric(df['value'], errors='coerce').astype('float32')
                    processed_df = self.preprocess_v3_frame(df)
                    batch_dfs.append(processed_df)
                    logging.info(f"Successfully processed {filename}")
                    del df
                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
                    continue
            if batch_dfs:
                batch_combined = pd.concat(batch_dfs).sort_index()
                all_dfs.append(batch_combined)
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size}")
                del batch_dfs, batch_combined

        if not all_dfs:
            raise RuntimeError("No valid data processed from any CSV files.")
        combined_df = pd.concat(all_dfs).sort_index()
        logging.info(f"Combined data shape after v3 preprocessing: {combined_df.shape}")
        del all_dfs
        return combined_df

    def create_windows(self, scaled_values, time_features):
        """
        Creates windowed datasets of (X, y) where:
          X shape = (n_windows, 96, 8)  — [scaled_load, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend]
          y shape = (n_windows, 96, 1)  — [scaled_load only]
        
        The input window is the PREVIOUS 96 steps.
        The output window is the NEXT 96 steps.
        Time features for the OUTPUT window are concatenated into X so the model
        knows what time/day it is predicting for.
        
        IMPORTANT: X contains the previous day's load (96 values) + the NEXT day's time features.
        This way the model receives:
            - What happened yesterday (load history)
            - What time period it's predicting (time features for tomorrow)
        
        Args:
            scaled_values: 1D array of MinMaxScaler-scaled load values, shape (N,)
            time_features: 2D array of time features, shape (N, 7)
        
        Returns:
            X: np.ndarray of shape (n_windows, 96, 8)
            y: np.ndarray of shape (n_windows, 96, 1)
        """
        logging.info("Creating training windows with time features...")
        scaled_values = scaled_values.reshape(-1)
        total_len = len(scaled_values)
        total_windows = total_len - self.seq_length - self.forecast_horizon + 1

        if total_windows <= 0:
            raise ValueError(f"Not enough data for windows: {total_len} data points, need at least {self.seq_length + self.forecast_horizon}")

        logging.info(f"Creating {total_windows} windows from {total_len} data points")

        X = np.zeros((total_windows, self.seq_length, self.n_input_features), dtype=np.float32)
        y = np.zeros((total_windows, self.forecast_horizon, 1), dtype=np.float32)

        batch_size = min(10000, total_windows)
        for batch_start in range(0, total_windows, batch_size):
            batch_end = min(batch_start + batch_size, total_windows)
            for i in range(batch_start, batch_end):
                # Input: previous day's scaled load values
                load_input = scaled_values[i:i + self.seq_length]
                # Time features for the OUTPUT/prediction window (next day)
                tf_output = time_features[i + self.seq_length:i + self.seq_length + self.forecast_horizon]
                
                # X[:, :, 0] = scaled load from input window
                # X[:, :, 1:8] = time features from OUTPUT window (what we're predicting for)
                X[i, :, 0] = load_input
                X[i, :, 1:] = tf_output
                
                # y = scaled load for the output window
                y[i, :, 0] = scaled_values[i + self.seq_length:i + self.seq_length + self.forecast_horizon]

            if total_windows > 50000:
                logging.info(f"Processed {batch_end}/{total_windows} windows...")

        logging.info(f"Window creation completed. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def create_tf_dataset(self, X, y, is_training=True, cache_file=None):
        """Creates an optimized tf.data.Dataset with file-based caching."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if is_training:
            dataset = dataset.map(
                lambda x, y: self._augment_data(x, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            if cache_file:
                dataset = dataset.cache(filename=cache_file)
                logging.info(f"Training dataset using file-based cache: {cache_file}")
            else:
                dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
        else:
            if cache_file:
                dataset = dataset.cache(filename=cache_file)
            else:
                dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        if is_training:
            dataset = dataset.map(
                lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return dataset

    def _augment_data(self, x, y):
        """
        Data augmentation for time series.
        Only augments the load channel (channel 0), time features are deterministic.
        """
        # Gaussian noise on load channel only
        noise_std = 0.01
        load_channel = x[:, 0:1]  # shape (96, 1)
        noise = tf.random.normal(shape=tf.shape(load_channel), mean=0.0, stddev=noise_std, dtype=load_channel.dtype)
        load_augmented = load_channel + noise

        # Amplitude scaling on load channel only
        scale_factor = tf.random.uniform(shape=[], minval=0.95, maxval=1.05, dtype=load_channel.dtype)
        load_augmented = load_augmented * scale_factor
        load_augmented = tf.clip_by_value(load_augmented, -2.0, 2.0)

        # Reconstruct x with augmented load + original time features
        time_features = x[:, 1:]  # shape (96, 7)
        x_out = tf.concat([load_augmented, time_features], axis=-1)
        return x_out, y


# =============================================================================
# 4. OPTIMIZED LSTM MODEL CLASS (MODIFIED — n_features=8)
# =============================================================================
class OptimizedLSTMForecaster:
    """
    DIRECT MANY-TO-MANY LSTM FORECASTER
    Input: (batch_size, 96, 8) — load + 7 time features
    Output: (batch_size, 96, 1) — predicted load only
    """
    def __init__(self, seq_length=96, n_features=8, forecast_horizon=96):
        self.seq_length = seq_length
        self.n_features = n_features  # 8: load + 7 time features
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.history = None
        self.train_loss_metric = None
        self.val_loss_metric = None

    def build_model(self, strategy):
        """
        DIRECT MANY-TO-MANY ARCHITECTURE
        Input: (batch_size, 96, 8) — load + 7 cyclical time features
        Output: (batch_size, 96, 1) — predicted load values
        """
        lstm_units_encoder = 256
        lstm_units_decoder = 128
        dropout_rate = 0.3
        learning_rate = 1e-4
        cnn_filters = 128

        logging.info("Building DIRECT Many-to-Many CNN-LSTM model (LOAD MODEL with time features)...")
        logging.info(f"Input shape: (batch, {self.seq_length}, {self.n_features})")
        logging.info(f"Output shape: (batch, {self.forecast_horizon}, 1)")

        with strategy.scope():
            inputs = tf.keras.layers.Input(shape=(self.seq_length, self.n_features))

            # CNN Encoder
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters * 2, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

            # LSTM Encoder
            encoder_output = tf.keras.layers.LSTM(
                lstm_units_encoder, return_sequences=False,
                kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform'
            )(x)
            encoder_output = tf.keras.layers.Dropout(dropout_rate)(encoder_output)

            # Decoder
            decoder_input = tf.keras.layers.RepeatVector(self.forecast_horizon)(encoder_output)
            x = tf.keras.layers.LSTM(
                lstm_units_decoder, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform'
            )(decoder_input)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.LSTM(
                lstm_units_decoder // 2, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform'
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.5)(x)

            # Output: predict load only (1 feature)
            outputs = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, activation='linear', dtype='float32'),
                name='output_layer'
            )(x)
            outputs = ClipByValue(-3.0, 3.0, name="soft_output_clipping")(outputs)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, clipnorm=1.0,
                beta_1=0.9, beta_2=0.999, epsilon=1e-7
            )
            self.loss_fn = tf.keras.losses.Huber(delta=1.0)
            self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

            self.model.compile(
                optimizer=self.optimizer, loss=self.loss_fn,
                metrics=[tf.keras.metrics.MeanAbsoluteError()],
                steps_per_execution=10
            )

            def build_optimizer_vars():
                dummy_input = tf.zeros((1, self.seq_length, self.n_features), dtype=tf.float32)
                dummy_output = tf.zeros((1, self.forecast_horizon, 1), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    dummy_pred = self.model(dummy_input, training=True)
                    dummy_loss = self.loss_fn(dummy_output, dummy_pred)
                trainable_vars = self.model.trainable_variables
                dummy_grads = tape.gradient(dummy_loss, trainable_vars)
                self.optimizer.apply_gradients(zip(dummy_grads, trainable_vars))
                return dummy_loss
            strategy.run(build_optimizer_vars)
            logging.info("Optimizer variables initialized successfully.")

        self.model.summary(print_fn=logging.info)
        logging.info(f"Model output shape: (batch_size, {self.forecast_horizon}, 1)")
        return self.model

    def build_tunable_model(self, hp, strategy):
        """Builds a tunable SOTA CNN-LSTM with Attention model for Keras Tuner."""
        logging.info("Building tunable CNN-LSTM with Attention (LOAD MODEL, 8 input features)...")

        with strategy.scope():
            lstm_units_1 = hp.Int('lstm_units_1', min_value=256, max_value=1024, step=256)
            lstm_units_2 = hp.Int('lstm_units_2', min_value=128, max_value=512, step=128)
            lstm_units_3 = hp.Int('lstm_units_3', min_value=64, max_value=256, step=64)
            dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=128)
            dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
            cnn_filters = hp.Int('cnn_filters', min_value=64, max_value=256, step=64)
            attention_heads = hp.Int('attention_heads', min_value=4, max_value=12, step=2)

            logging.info(f"Trial HPs: CNN({cnn_filters}) -> LSTM({lstm_units_1}->{lstm_units_2}->{lstm_units_3}) -> Attn({attention_heads}) -> Dense({dense_units})")

            inputs = tf.keras.layers.Input(shape=(self.seq_length, self.n_features))

            # CNN layers
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters * 2, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters * 2, kernel_size=5, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

            # LSTM layers
            x = tf.keras.layers.LSTM(lstm_units_1, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.LSTM(lstm_units_2, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.LSTM(lstm_units_3, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

            # Multi-Head Attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=attention_heads, key_dim=lstm_units_3 // attention_heads,
                dropout=dropout_rate * 0.5
            )(x, x)
            x = tf.keras.layers.Add()([x, attention_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # Temporal aggregation
            attention_weights = tf.keras.layers.Dense(1, activation='softmax', name='temporal_attention_tunable')(x)
            weighted_features = tf.keras.layers.Multiply()([x, attention_weights])
            x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='temporal_aggregation_tunable')(weighted_features)

            # Dense layers
            x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.5)(x)
            x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.4)(x)
            x = tf.keras.layers.Dense(dense_units // 2, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

            # Output layer: forecast_horizon outputs → reshape to (96, 1)
            outputs = tf.keras.layers.Dense(self.forecast_horizon, activation='linear', dtype='float32')(x)
            outputs = ClipByValue(-3.0, 3.0, name="soft_output_clipping_tunable")(outputs)
            outputs = tf.keras.layers.Reshape((self.forecast_horizon, 1), name='forecast_horizon_reshape')(outputs)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, clipnorm=1.0,
                beta_1=0.9, beta_2=0.999, epsilon=1e-7
            )
            self.loss_fn = tf.keras.losses.Huber(delta=1.0)
            self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

            self.model.compile(
                optimizer=self.optimizer, loss=self.loss_fn,
                metrics=[tf.keras.metrics.MeanAbsoluteError()],
                steps_per_execution=10
            )

            def build_optimizer_vars():
                dummy_input = tf.zeros((1, self.seq_length, self.n_features), dtype=tf.float32)
                dummy_output = tf.zeros((1, self.forecast_horizon, 1), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    dummy_pred = self.model(dummy_input, training=True)
                    dummy_loss = self.loss_fn(dummy_output, dummy_pred)
                trainable_vars = self.model.trainable_variables
                dummy_grads = tape.gradient(dummy_loss, trainable_vars)
                self.optimizer.apply_gradients(zip(dummy_grads, trainable_vars))
                return dummy_loss
            strategy.run(build_optimizer_vars)
            logging.info("Tunable model built successfully.")
            total_params = self.model.count_params()
            logging.info(f"Model complexity: {total_params:,} total parameters")

        return self.model

    @tf.function
    def _train_step(self, x_batch, y_batch):
        """Direct training step — predicts all 96 steps in one forward pass."""
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)  # (batch, 96, 1)
            horizon_weights = tf.exp(-tf.range(self.forecast_horizon, dtype=tf.float32) / self.forecast_horizon)
            horizon_weights = horizon_weights / tf.reduce_sum(horizon_weights)
            per_step_loss = self.loss_fn(y_batch, predictions)
            loss = tf.reduce_mean(horizon_weights * per_step_loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)

    @tf.function
    def _test_step(self, x_batch, y_batch):
        """Validation step."""
        predictions = self.model(x_batch, training=False)
        horizon_weights = tf.exp(-tf.range(self.forecast_horizon, dtype=tf.float32) / self.forecast_horizon)
        horizon_weights = horizon_weights / tf.reduce_sum(horizon_weights)
        per_step_loss = self.loss_fn(y_batch, predictions)
        loss = tf.reduce_mean(horizon_weights * per_step_loss)
        self.val_loss_metric.update_state(loss)

    def fit(self, train_dataset, validation_dataset, epochs=100, patience=15):
        """Training with early stopping, LR scheduling, and checkpointing."""
        logging.info(f"Starting DIRECT many-to-many training for {epochs} epochs...")
        self.history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        lr_reduction_patience = 10
        lr_reduction_counter = 0
        min_lr = 1e-7

        checkpoint_dir = os.path.join(os.getcwd(), 'model_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')

        for epoch in range(epochs):
            logging.info(f"\nEpoch {epoch+1}/{epochs}")
            if epoch == 0:
                warmup_lr = self.optimizer.learning_rate.numpy() * 0.1
                self.optimizer.learning_rate.assign(warmup_lr)
                logging.info(f"Warmup phase: LR = {warmup_lr:.6f}")
            elif epoch == 5:
                full_lr = self.optimizer.learning_rate.numpy() * 10
                self.optimizer.learning_rate.assign(full_lr)
                logging.info(f"Warmup complete: LR = {full_lr:.6f}")
            elif epoch > 20:
                progress = (epoch - 20) / (epochs - 20)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                new_lr = max(self.optimizer.learning_rate.numpy() * cosine_decay, min_lr)
                self.optimizer.learning_rate.assign(new_lr)

            self.train_loss_metric.reset_state()
            self.val_loss_metric.reset_state()

            train_steps = 0
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                self._train_step(x_batch, y_batch)
                train_steps += 1
                if step % 100 == 0:
                    logging.info(f"Epoch {epoch+1}, Step {step}, Training...")
                    try:
                        gpus = tf.config.list_physical_devices('GPU')
                        if gpus:
                            for i, gpu in enumerate(gpus):
                                gpu_memory = tf.config.experimental.get_memory_info(f'GPU:{i}')
                                if gpu_memory:
                                    logging.info(f"GPU {i} Memory: {gpu_memory['current'] / 1024**3:.2f} GB / {gpu_memory['peak'] / 1024**3:.2f} GB")
                    except:
                        pass

            train_loss = self.train_loss_metric.result().numpy()
            self.history['loss'].append(train_loss)
            logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.6f} ({train_steps} steps)")

            for x_batch_val, y_batch_val in validation_dataset:
                self._test_step(x_batch_val, y_batch_val)
            val_loss = self.val_loss_metric.result().numpy()
            self.history['val_loss'].append(val_loss)
            logging.info(f"Epoch {epoch+1} Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                lr_reduction_counter = 0
                try:
                    self.model.save_weights(best_model_path)
                    logging.info(f"Best val loss: {best_val_loss:.6f} — saved")
                except Exception as e:
                    logging.warning(f"Failed to save weights: {e}")
            else:
                patience_counter += 1
                lr_reduction_counter += 1
                logging.info(f"No improvement for {patience_counter} epochs")
                if lr_reduction_counter >= lr_reduction_patience:
                    current_lr = self.optimizer.learning_rate.numpy()
                    if current_lr > min_lr:
                        new_lr = current_lr * 0.5
                        self.optimizer.learning_rate.assign(new_lr)
                        logging.info(f"LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                        lr_reduction_counter = 0

            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                try:
                    self.model.load_weights(best_model_path)
                    logging.info("Best weights restored.")
                except Exception as e:
                    logging.warning(f"Failed to load best weights: {e}")
                break

        return self.model

    def evaluate_model(self, test_dataset):
        """Evaluate model and return per-step metrics."""
        logging.info("Evaluating model on test set (DIRECT forecasting)...")
        all_forecasts = []
        all_actuals = []
        for x_batch, y_batch in test_dataset:
            predictions = self.model(x_batch, training=False)
            all_forecasts.append(predictions.numpy())
            all_actuals.append(y_batch.numpy())

        y_pred_full = np.concatenate(all_forecasts, axis=0)
        y_true_full = np.concatenate(all_actuals, axis=0)
        y_pred_flat = y_pred_full.flatten()
        y_true_flat = y_true_full.flatten()

        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + epsilon))) * 100

        metrics = {
            'MSE': float(mse), 'RMSE': float(rmse),
            'MAE': float(mae), 'MAPE': float(mape)
        }

        logging.info("=" * 60)
        logging.info("DIRECT FORECASTING PERFORMANCE (LOAD MODEL)")
        logging.info("=" * 60)
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        per_step_mae = np.mean(np.abs(y_true_full - y_pred_full), axis=(0, 2))
        logging.info(f"\nPer-Horizon MAE:")
        logging.info(f"  Steps 1-5:   {[f'{x:.4f}' for x in per_step_mae[:5]]}")
        logging.info(f"  Steps 46-50: {[f'{x:.4f}' for x in per_step_mae[45:50]]}")
        logging.info(f"  Steps 92-96: {[f'{x:.4f}' for x in per_step_mae[-5:]]}")

        early_mae = np.mean(per_step_mae[:24])
        mid_mae = np.mean(per_step_mae[24:72])
        late_mae = np.mean(per_step_mae[72:])
        logging.info(f"\nError by Period:")
        logging.info(f"  Early (0-6h):   {early_mae:.4f}")
        logging.info(f"  Middle (6-18h): {mid_mae:.4f}")
        logging.info(f"  Late (18-24h):  {late_mae:.4f}")

        return metrics, y_pred_flat, y_true_flat, per_step_mae


# =============================================================================
# 5. KERAS TUNER INTEGRATION CLASS
# =============================================================================
class TimeSeriesHyperModel(keras_tuner.HyperModel):
    def __init__(self, seq_length=96, n_features=8, forecast_horizon=96, strategy=None):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.strategy = strategy
        self.forecaster = None

    def build(self, hp):
        self.forecaster = OptimizedLSTMForecaster(
            seq_length=self.seq_length,
            n_features=self.n_features,
            forecast_horizon=self.forecast_horizon
        )
        self.forecaster.build_tunable_model(hp, self.strategy)
        return self.forecaster.model

    def fit(self, hp, model, train_dataset, **kwargs):
        validation_dataset = kwargs.get('validation_data')
        if validation_dataset is None:
            raise ValueError("validation_data must be provided")
        epochs = kwargs.get('epochs', 50)
        patience = kwargs.get('patience', 10)
        self.forecaster.fit(train_dataset, validation_dataset, epochs=epochs, patience=patience)
        best_val_loss = float(min(self.forecaster.history['val_loss']))
        return {"val_loss": best_val_loss}


# =============================================================================
# 6. VISUALIZATION CLASS
# =============================================================================
class TimeSeriesVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        plt.style.use('default')

    def plot_data_overview(self, df):
        fig = df['value_filled'].plot(figsize=(15, 5), title='Time Series Overview (v3 Preprocessed)').get_figure()
        plt.savefig(os.path.join(self.output_dir, 'data_overview.png'))
        plt.close(fig)

    def plot_training_history(self, history):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close(fig)

    def plot_predictions(self, y_true, y_pred, n_samples=500):
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Advanced Model Performance Analysis (Load Model)', fontsize=16, fontweight='bold')

        axes[0, 0].plot(y_true[:n_samples], label='Actual', linewidth=1.5, color='blue')
        axes[0, 0].plot(y_pred[:n_samples], label='Predicted', linewidth=1.5, alpha=0.8, color='red')
        axes[0, 0].set_title(f'Time Series Comparison (First {n_samples} Samples)')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Load Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(y_true[:n_samples], y_pred[:n_samples], alpha=0.6, s=20)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_title('Predicted vs Actual Values')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        residuals = y_true[:n_samples] - y_pred[:n_samples]
        axes[1, 0].plot(residuals, color='green', linewidth=1)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual Values')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'advanced_predictions_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        analysis_data = {
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals)),
            'residuals_mae': float(np.mean(np.abs(residuals))),
            'correlation': float(np.corrcoef(y_true[:n_samples], y_pred[:n_samples])[0, 1])
        }
        analysis_path = os.path.join(self.output_dir, 'prediction_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=4)
        logging.info(f"Prediction analysis saved to {analysis_path}")

    def plot_error_by_horizon(self, per_step_mae):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, len(per_step_mae) + 1), per_step_mae, marker='o', linestyle='-', color='purple')
        ax.set_title('Mean Absolute Error (MAE) vs. Forecast Horizon')
        ax.set_xlabel('Forecast Step (Horizon)')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_xticks(range(0, len(per_step_mae) + 1, 12))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_by_horizon.png'), dpi=300)
        plt.close(fig)
        logging.info("Error by horizon plot saved.")


# =============================================================================
# 7. HYPERPARAMETER OPTIMIZATION
# =============================================================================
def run_hyperparameter_optimization(train_dataset, validation_dataset, strategy,
                                    output_dir, n_features=8, max_trials=40,
                                    epochs_per_trial=30):
    logging.info("=" * 50)
    logging.info("HYPERPARAMETER OPTIMIZATION (LOAD MODEL, 8-feature input)")
    logging.info("=" * 50)

    keras_tuner_dir = os.path.join(output_dir, 'keras_tuner')
    if os.path.exists(keras_tuner_dir):
        import shutil
        shutil.rmtree(keras_tuner_dir)

    hypermodel = TimeSeriesHyperModel(
        seq_length=96, n_features=n_features,
        forecast_horizon=96, strategy=strategy
    )

    tuner = keras_tuner.RandomSearch(
        hypermodel,
        objective=keras_tuner.Objective('val_loss', direction='min'),
        max_trials=max_trials, executions_per_trial=1,
        directory=os.path.join(output_dir, 'keras_tuner'),
        project_name='load_model_forecasting',
        overwrite=True, seed=42
    )

    logging.info("Starting RandomSearch...")
    try:
        tuner.search(
            train_dataset, validation_data=validation_dataset,
            epochs=epochs_per_trial, verbose=1
        )
        best_hp = tuner.get_best_hyperparameters(1)[0]
        logging.info("\nBest hyperparameters:")
        for param, value in best_hp.values.items():
            logging.info(f"  {param}: {value}")

        best_hp_path = os.path.join(output_dir, 'best_hyperparameters.json')
        with open(best_hp_path, 'w') as f:
            json.dump(best_hp.values, f, indent=4)
        logging.info(f"Best HPs saved to {best_hp_path}")

        best_model = tuner.hypermodel.build(best_hp)
        return best_model, best_hp

    except Exception as e:
        logging.error(f"HP optimization failed: {e}")
        logging.info("Falling back to default hyperparameters...")
        default_forecaster = OptimizedLSTMForecaster(
            seq_length=96, n_features=n_features, forecast_horizon=96
        )
        default_model = default_forecaster.build_model(strategy)
        default_hp = {
            'lstm_units_1': 384, 'lstm_units_2': 192, 'lstm_units_3': 128,
            'dense_units': 256, 'dropout_rate': 0.3,
            'learning_rate': 4.77e-05
        }
        default_hp_path = os.path.join(output_dir, 'default_hyperparameters.json')
        with open(default_hp_path, 'w') as f:
            json.dump(default_hp, f, indent=4)
        return default_model, default_hp


# =============================================================================
# 8. MAIN EXECUTION FUNCTION
# =============================================================================
def main(args):
    setup_logging(args.logs_dir)

    if 'SLURM_JOB_ID' in os.environ:
        logging.info("=" * 60)
        logging.info("SLURM ENVIRONMENT DETECTED")
        logging.info(f"Job ID: {os.environ['SLURM_JOB_ID']}")
        logging.info(f"Job Name: {os.environ.get('SLURM_JOB_NAME', 'N/A')}")
        logging.info(f"GPUs: {os.environ.get('SLURM_GPUS_PER_NODE', 'N/A')}")
        logging.info("=" * 60)

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("cpu_only: forcing CPU-only execution")
    else:
        configure_gpu()

    if args.cpu_only:
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        logging.info("CPU-only strategy initialized")
    else:
        try:
            strategy = tf.distribute.MirroredStrategy()
            logging.info(f"Multi-GPU: {strategy.num_replicas_in_sync} devices")
        except Exception as e:
            logging.warning(f"Multi-GPU failed: {e}. Falling back to single GPU.")
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    logging.info(f"Global batch size: {global_batch_size}")

    n_input_features = 8  # 1 (load) + 7 (time features)

    data_processor = TimeSeriesDataProcessor(
        seq_length=96, forecast_horizon=96,
        batch_size=global_batch_size,
        dataset_kind=args.dataset_kind,
        n_input_features=n_input_features
    )
    visualizer = TimeSeriesVisualizer(args.output_dir)

    # =========================================================================
    # STEP 1: DATA LOADING AND PREPROCESSING (v3 only, NO differencing)
    # =========================================================================
    logging.info("\n" + "=" * 50)
    logging.info("STEP 1: DATA LOADING AND PREPROCESSING")
    logging.info("=" * 50)
    logging.info("v3 Preprocessing: gap classification + hierarchical fill")
    logging.info("NO DIFFERENCING — model predicts absolute load values")
    logging.info("7 cyclical time features added from timestamps")
    logging.info("=" * 50)

    df = data_processor.load_and_process_data(args.data_dir)
    log_memory_usage("after data loading")
    visualizer.plot_data_overview(df)

    # Save preprocessed series
    preprocessed_path = os.path.join(args.output_dir, 'preprocessed_value_filled.csv')
    df.to_csv(preprocessed_path, index_label='time')
    logging.info(f"Preprocessed data saved to {preprocessed_path}")

    data_values = df['value_filled'].values
    timestamps = df.index  # DatetimeIndex preserved from v3 preprocessing

    del df
    gc.collect()
    log_memory_usage("after clearing dataframe")

    # Handle remaining NaNs
    if np.isnan(data_values).any():
        nan_count = np.isnan(data_values).sum()
        logging.warning(f"Found {nan_count} remaining NaN values. Forward-filling...")
        data_values = pd.Series(data_values).fillna(method='ffill').fillna(method='bfill').values

    # =========================================================================
    # COMPUTE TIME FEATURES (NEW)
    # =========================================================================
    logging.info("\n" + "=" * 50)
    logging.info("COMPUTING CYCLICAL TIME FEATURES")
    logging.info("=" * 50)
    logging.info("Features: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend")

    time_features = compute_time_features(timestamps)
    logging.info(f"Time features shape: {time_features.shape}")
    logging.info(f"Time features sample (first 3 rows):\n{time_features[:3]}")

    # =========================================================================
    # SPLIT DATA BEFORE SCALING (prevent data leakage)
    # =========================================================================
    train_ratio, val_ratio = 0.8, 0.1
    train_idx = int(len(data_values) * train_ratio)
    val_idx = int(len(data_values) * (train_ratio + val_ratio))

    train_data = data_values[:train_idx]
    val_data = data_values[train_idx:val_idx]
    test_data = data_values[val_idx:]

    train_tf = time_features[:train_idx]
    val_tf = time_features[train_idx:val_idx]
    test_tf = time_features[val_idx:]

    logging.info(f"Data split — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    logging.info(f"Train range: [{train_data.min():.2f}, {train_data.max():.2f}]")

    # =========================================================================
    # SCALE LOAD VALUES (fit scaler on train only)
    # =========================================================================
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    val_scaled = scaler.transform(val_data.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.reshape(-1, 1))

    logging.info(f"Scaler fitted on RAW load values: min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f}")
    logging.info("CRITICAL: These min/max values are for RAW load data (NOT differenced)")

    # =========================================================================
    # CREATE WINDOWS WITH TIME FEATURES
    # =========================================================================
    X_train, y_train = data_processor.create_windows(train_scaled, train_tf)
    log_memory_usage("after train windows")
    X_val, y_val = data_processor.create_windows(val_scaled, val_tf)
    X_test, y_test = data_processor.create_windows(test_scaled, test_tf)
    log_memory_usage("after all windows")

    logging.info(f"Train windows: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    del train_scaled, val_scaled, test_scaled
    gc.collect()

    # Create TF datasets
    train_cache = os.path.join(args.output_dir, 'train_cache')
    val_cache = os.path.join(args.output_dir, 'val_cache')
    test_cache = os.path.join(args.output_dir, 'test_cache')

    train_dataset = data_processor.create_tf_dataset(X_train, y_train, cache_file=train_cache)
    val_dataset = data_processor.create_tf_dataset(X_val, y_val, is_training=False, cache_file=val_cache)
    test_dataset = data_processor.create_tf_dataset(X_test, y_test, is_training=False, cache_file=test_cache)

    # =========================================================================
    # STEP 2: MODEL BUILDING AND TRAINING
    # =========================================================================
    if args.use_tuner:
        logging.info("\n" + "=" * 50)
        logging.info("STEP 2: HYPERPARAMETER OPTIMIZATION (LOAD MODEL)")
        logging.info("=" * 50)

        best_model, best_hp = run_hyperparameter_optimization(
            train_dataset=train_dataset, validation_dataset=val_dataset,
            strategy=strategy, output_dir=args.output_dir,
            n_features=n_input_features,
            max_trials=args.max_trials, epochs_per_trial=args.epochs_per_trial
        )

        model_forecaster = OptimizedLSTMForecaster(
            seq_length=96, n_features=n_input_features, forecast_horizon=96
        )
        model_forecaster.model = best_model

        model_path = os.path.join(args.output_dir, 'best_tuned_model_savedmodel')
        best_model.export(model_path)
        logging.info(f"Best tuned model exported to: {model_path}")

    else:
        logging.info("\n" + "=" * 50)
        logging.info("STEP 2: DIRECT MANY-TO-MANY MODEL TRAINING (LOAD MODEL)")
        logging.info("=" * 50)
        logging.info(f"Input: (batch, 96, {n_input_features}) — load + 7 time features")
        logging.info("Output: (batch, 96, 1) — predicted load only")
        logging.info("NO differencing. Model predicts absolute load values directly.")
        logging.info("=" * 50)

        model_forecaster = OptimizedLSTMForecaster(
            seq_length=96, n_features=n_input_features, forecast_horizon=96
        )
        model_forecaster.build_model(strategy)

        start_time = time.time()
        model_forecaster.fit(train_dataset, val_dataset, epochs=args.epochs, patience=args.patience)
        training_time = time.time() - start_time
        logging.info(f"\nTraining completed in {training_time:.2f} seconds")

        visualizer.plot_training_history(model_forecaster.history)

        model_path = os.path.join(args.output_dir, 'final_model_savedmodel')
        model_forecaster.model.export(model_path)
        logging.info(f"Model exported to: {model_path}")

        if args.export_cpu:
            cpu_export_dir = os.path.join(args.output_dir, "final_model_savedmodel_cpu")
            keras_backup_path = os.path.join(args.output_dir, "final_model_for_cpu_export.keras")
            model_forecaster.model.save(keras_backup_path)
            logging.info(f"Saved Keras model for CPU export: {keras_backup_path}")
            script_path = os.path.join(os.path.dirname(__file__), "export_cpu_savedmodel.py")
            if os.path.exists(script_path):
                cmd = [
                    sys.executable, script_path,
                    "--keras_path", keras_backup_path,
                    "--export_dir", cpu_export_dir,
                    "--seq_len", str(args.seq_len),
                    "--n_features", str(n_input_features),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                env["TF_ENABLE_ONEDNN_OPTS"] = "0"
                env["TF_USE_CUDNN"] = "0"
                logging.info(f"Launching CPU export: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, env=env)
                logging.info(f"CPU-compatible SavedModel exported to: {cpu_export_dir}")
            else:
                logging.warning(f"CPU export script not found at {script_path}, skipping.")

    # =========================================================================
    # STEP 3: MODEL EVALUATION
    # =========================================================================
    logging.info("\n" + "=" * 50)
    logging.info("STEP 3: MODEL EVALUATION")
    logging.info("=" * 50)

    metrics, y_pred_scaled, y_true_scaled, per_step_mae = model_forecaster.evaluate_model(test_dataset)

    # Inverse transform to get actual load values
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_error_by_horizon(per_step_mae)

    metrics_path = os.path.join(args.output_dir, 'performance_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

    # =========================================================================
    # STEP 4: EXPORT NORMALIZATION PARAMETERS
    # =========================================================================
    logging.info("\n" + "=" * 50)
    logging.info("STEP 4: EXPORTING NORMALIZATION PARAMETERS")
    logging.info("=" * 50)

    scaler_params = {
        'max_value': float(scaler.data_max_[0]),
        'min_value': float(scaler.data_min_[0]),
        'feature_range': [-1, 1],
        'normalization_method': 'MinMaxScaler',
        'description': 'Scaler fitted on RAW (non-differenced) load values from training data only.',
        'model_type': 'load_model_with_time_features',
        'n_input_features': n_input_features,
        'time_features': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend'],
        'differencing': 'NONE — model predicts absolute load values directly',
        'usage_instructions': {
            'input_normalization': f'scaled = (load_value - {scaler.data_min_[0]}) / ({scaler.data_max_[0]} - {scaler.data_min_[0]}) * 2 - 1',
            'output_denormalization': f'load_value = (predicted + 1) / 2 * ({scaler.data_max_[0]} - {scaler.data_min_[0]}) + {scaler.data_min_[0]}',
        },
        'training_data_stats': {
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'test_samples': len(test_data),
            'raw_data_range': [float(train_data.min()), float(train_data.max())],
        }
    }

    params_path = os.path.join(args.output_dir, 'scaler_params.json')
    with open(params_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)

    simple_params = {
        'max_value': float(scaler.data_max_[0]),
        'min_value': float(scaler.data_min_[0])
    }
    simple_params_path = os.path.join(args.output_dir, 'normalization_params_simple.json')
    with open(simple_params_path, 'w') as f:
        json.dump(simple_params, f, indent=4)

    logging.info(f"Scaler params saved to {params_path}")
    logging.info(f"Simple params saved to {simple_params_path}")
    logging.info(f"  data_max (RAW): {scaler_params['max_value']}")
    logging.info(f"  data_min (RAW): {scaler_params['min_value']}")
    logging.info(f"  n_input_features: {n_input_features}")
    logging.info("")
    logging.info("CRITICAL: These are min/max of RAW load values (NOT differenced).")
    logging.info("NO differencing parameters — model predicts absolute values directly.")
    logging.info("Time features are computed from timestamps at inference time.")
    logging.info("Update config.ini with data_max and data_min for the implementation script.")
    logging.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Model Trainer — CNN-LSTM with Time Features")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data CSVs.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save models and metrics.')
    parser.add_argument('--logs_dir', type=str, required=True, help='Directory to save logs.')
    parser.add_argument('--dataset_kind', type=str, default='electric',
                        choices=['electric', 'solar'],
                        help='Dataset type: electric treats zeros as missing.')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU.')
    parser.add_argument('--use_tuner', action='store_true', help='Use Keras Tuner for HP optimization.')
    parser.add_argument('--max_trials', type=int, default=40, help='Max HP trials.')
    parser.add_argument('--epochs_per_trial', type=int, default=30, help='Epochs per HP trial.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument("--export_cpu", action="store_true",
                        help="Also export CPU-compatible SavedModel.")
    parser.add_argument("--seq_len", type=int, default=96, help="Sequence length (96 = 24h at 15-min).")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force CPU-only execution.")
    args = parser.parse_args()
    main(args)
