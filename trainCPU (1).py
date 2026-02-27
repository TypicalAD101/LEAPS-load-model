# THIS FILE IS train_SOLAR_FAST_TOBEUSED.py
# LSTM Time Series Forecasting Model - DIRECT MANY-TO-MANY VERSION
# - Many-to-many architecture: Input 96 steps â†’ Output 96 steps in ONE forward pass
# - NO recursion, NO error accumulation, NO cuDNN compatibility issues
# - Daily seasonal differencing (lag=96) for strong seasonality handling
# - MinMaxScaler normalization with production export
# - Multi-GPU training with mixed precision
# - 50-100x faster training than recursive approach

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
import sys
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
        
        # Log GPU memory if available
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

# REMOVED: get_sampling_probability() function
# No longer needed for direct many-to-many forecasting (no scheduled sampling)

def configure_gpu():
    """Configure GPU for optimal HPC performance with multi-GPU support."""
    # Check for SLURM environment variables and respect them
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices:
        logging.info(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices}")
    
    # Check for TensorFlow GPU growth setting
    tf_force_gpu_growth = os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'false').lower() == 'true'
    if tf_force_gpu_growth:
        logging.info("TF_FORCE_GPU_ALLOW_GROWTH is enabled")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                # Respect TF_FORCE_GPU_ALLOW_GROWTH environment variable
                if not tf_force_gpu_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Memory growth enabled for GPU {gpu}")
                
                # For full A100-80GB, optimize memory usage
                try:
                    # Try to set a higher memory limit for A100 80GB
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=75000)]
                    )
                    logging.info(f"Memory limit set to 75GB for GPU {gpu}")
                except Exception as e:
                    logging.warning(f"Could not set memory limit for GPU {gpu}: {e}")
                    # Fallback to default memory management
            logging.info(f"GPU configured: {len(physical_devices)} GPU(s) available.")
            
            # Enhanced mixed precision for A100 and other modern GPUs
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logging.info(f"Mixed precision enabled: {policy.name}")
            except Exception as e:
                logging.warning(f"Mixed precision not available: {e}")
                # Fallback to float32
                policy = tf.keras.mixed_precision.Policy('float32')
                tf.keras.mixed_precision.set_global_policy(policy)
                logging.info(f"Using float32 precision: {policy.name}")

        except RuntimeError as e:
            logging.error(f"GPU configuration error: {e}")
    else:
        logging.info("No GPU available, using CPU.")
    
    # Enhanced XLA compilation for HPC performance
    try:
        tf.config.optimizer.set_jit(True)
        logging.info("XLA compilation enabled for optimal performance.")
    except Exception as e:
        logging.warning(f"XLA compilation not available: {e}")
    
    # Enable device placement logging for debugging multi-GPU usage (only in debug mode)
    if os.environ.get('TF_DEBUG_DEVICE_PLACEMENT', 'false').lower() == 'true':
        tf.debugging.set_log_device_placement(True)
        logging.info("Device placement logging enabled for multi-GPU debugging")
    else:
        logging.info("Device placement logging disabled (set TF_DEBUG_DEVICE_PLACEMENT=true to enable)")
    
    # Additional HPC optimizations
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
    })
    logging.info("Advanced TensorFlow optimizations enabled for HPC performance")
    
    # Memory optimization settings for large models
    # Note: TF_FORCE_GPU_ALLOW_GROWTH=true is set in SLURM script
    # This Python code is not needed when using the environment variable approach
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    
    # Enable memory optimization
    tf.config.optimizer.set_experimental_options({
        "pin_to_host_optimization": False,  # Disable pinning to host memory
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
    })
    logging.info("Memory optimization settings enabled")
    
    # Log TensorFlow environment variables for debugging
    tf_vars = ['TF_GPU_THREAD_MODE', 'TF_GPU_THREAD_COUNT', 'TF_USE_CUDNN']
    
    logging.info("TensorFlow Environment Variables:")
    for var in tf_vars:
        value = os.environ.get(var, 'Not Set')
        logging.info(f"  {var}: {value}")
    
    # Note: NCCL variables are now controlled by SLURM scripts for optimal performance


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
# 2. ADVANCED v3 DATA PROCESSING CLASS
# =============================================================================
class TimeSeriesDataProcessor:
    """
    Handles data loading, cleaning, and preparation for the model using
    sophisticated v3 preprocessing with gap classification and hierarchical filling.
    """
    def __init__(self, seq_length=96, forecast_horizon=96, batch_size=1024, dataset_kind="electric"):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.dataset_kind = dataset_kind

    def classify_gaps(self, s, short_hours=7, points_per_hour=4):
        """Return boolean masks for short vs long gaps based on consecutive NaNs."""
        na = s.isna()
        run_id = (na != na.shift()).cumsum()
        run_len = na.groupby(run_id).transform('sum')
        short_limit = short_hours * points_per_hour  # 28 for 7h at 15-min
        short_mask = na & (run_len < short_limit)
        long_mask = na & (run_len >= short_limit)
        return short_mask, long_mask

    def preprocess_v3_frame(self, df, short_hours=7):
        """Apply v3 preprocessing to a single time-indexed frame with column 'value'."""
        cols = [c.strip().lower() for c in df.columns]
        time_col = df.columns[cols.index('time')] if 'time' in cols else None
        val_col = df.columns[cols.index('value')] if 'value' in cols else None
        if time_col is None or val_col is None:
            raise ValueError("Frame must contain 'time' and 'value' columns.")
        
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

        # Build series.
        # For electricity/load: treat zeros as missing for gap detection.
        # For solar: keep nighttime zeros as valid values.
        s = pd.to_numeric(df[val_col], errors='coerce')
        if self.dataset_kind != "solar":
            s = s.replace(0, np.nan)
        
        # Log initial statistics
        total_points = len(s)
        zero_points = (s == 0).sum()
        nan_points = s.isna().sum()
        logging.info(f"Initial data: {total_points} points, {zero_points} zeros, {nan_points} NaN values")

        # Assume 15-minute cadence for prev-day lag = 96
        points_per_hour = 4
        short_mask, long_mask = self.classify_gaps(s, short_hours=short_hours, points_per_hour=points_per_hour)
        
        # Log gap classification results
        short_gaps = short_mask.sum()
        long_gaps = long_mask.sum()
        logging.info(f"Gap classification: {short_gaps} short gaps (<{short_hours}h), {long_gaps} long gaps (â‰¥{short_hours}h)")

        # Short-gap candidates: D-1..D-7 same-time
        lags = [96, 2*96, 3*96, 4*96, 5*96, 6*96, 7*96]
        cand = pd.concat([s.shift(L) for L in lags], axis=1)
        cand.columns = [f"lag_{L}" for L in lags]
        prev_days_first = cand.bfill(axis=1).iloc[:, 0]

        # ToD medians (map by HH:MM); keep as Series aligned to s
        tod_key = pd.Series(s.index.strftime('%H:%M'), index=s.index)
        tod_medians = s.groupby(tod_key).median()
        tod_fill = tod_key.map(tod_medians)

        # Safety net
        locf = s.ffill()

        # Assemble short-gap fill
        short_fill = prev_days_first.copy()
        short_fill = short_fill.where(short_fill.notna(), tod_fill)
        short_fill = short_fill.where(short_fill.notna(), locf)

        after_short = s.copy()
        after_short.loc[short_mask] = short_fill.loc[short_mask]

        # Long-gap fill â†’ ToD median
        filled = after_short.copy()
        filled.loc[long_mask] = tod_fill.loc[long_mask]

        # Log final filling results
        remaining_nans = filled.isna().sum()
        logging.info(f"After v3 preprocessing: {remaining_nans} NaN values remain out of {total_points} total points")

        out = pd.DataFrame(index=df.index)
        out['value_filled'] = filled
        return out

    def load_and_process_data(self, data_dir):
        """Loads all CSVs from a directory, applies v3 preprocessing, and combines them."""
        logging.info(f"Loading and applying v3 preprocessing to all data from directory: {data_dir}")
        
        # Get list of CSV files first
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        logging.info(f"Found {len(csv_files)} CSV files to process")
        
        if not csv_files:
            raise RuntimeError("No CSV files found in data_dir OR No valid CSVs found with columns time/value.")
        
        batch_size = 5
        all_dfs = []
        
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            batch_dfs = []
            
            for filename in batch_files:
                file_path = os.path.join(data_dir, filename)
                logging.info(f"Processing {filename} with v3 preprocessing...")
                try:
                    # Read CSV with memory optimization
                    df = pd.read_csv(file_path)

                    # --- PATCH: normalize column names to 'time' and 'value' (for solar NREL files) ---
                    _cols = {c.strip().lower(): c for c in df.columns}
                    _rename = {}

                    # time column
                    if 'time' in _cols:
                        _rename[_cols['time']] = 'time'
                    else:
                        for _k in ('localtime', 'local_time', 'datetime', 'date_time', 'timestamp', 'date', 'dt'):
                            if _k in _cols:
                                _rename[_cols[_k]] = 'time'
                                break

                    # value column
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

                    # optional: cast value to float32 for memory savings
                    df['value'] = pd.to_numeric(df['value'], errors='coerce').astype('float32')
                    # --- END PATCH ---

                    processed_df = self.preprocess_v3_frame(df)
                    batch_dfs.append(processed_df)
                    logging.info(f"Successfully processed {filename}")
                    
                    # Clear individual dataframe to free memory
                    del df
                    
                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
                    continue
            
            # Combine batch and clear individual dataframes
            if batch_dfs:
                batch_combined = pd.concat(batch_dfs).sort_index()
                all_dfs.append(batch_combined)
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size}")
                
                # Clear batch dataframes to free memory
                del batch_dfs
                del batch_combined
        
        # Final combination
        if not all_dfs:
            raise RuntimeError("No valid data processed from any CSV files.")
        
        combined_df = pd.concat(all_dfs).sort_index()
        logging.info(f"Combined data shape after v3 preprocessing: {combined_df.shape}")
        
        # Clear intermediate dataframes
        del all_dfs
        
        return combined_df

    def create_windows(self, data):
        """
        Creates windowed datasets of (X, y) where X is 96 history steps
        and y is the next 96 future steps. This is required for the custom
        training loop's loss calculation.
        
        Memory-optimized version to prevent system OOM.
        """
        logging.info("Creating training windows with memory optimization...")
        data = data.reshape(-1, 1)  # Ensure data is 2D
        
        # Calculate total windows to pre-allocate arrays
        total_windows = len(data) - self.seq_length - self.forecast_horizon + 1
        logging.info(f"Creating {total_windows} windows from {len(data)} data points")
        
        # Pre-allocate arrays to avoid memory fragmentation
        X = np.zeros((total_windows, self.seq_length, 1), dtype=np.float32)
        y = np.zeros((total_windows, self.forecast_horizon, 1), dtype=np.float32)
        
        # Fill arrays in batches to manage memory
        batch_size = min(10000, total_windows)  # Process in smaller batches
        for batch_start in range(0, total_windows, batch_size):
            batch_end = min(batch_start + batch_size, total_windows)
            for i in range(batch_start, batch_end):
                X[i] = data[i:(i + self.seq_length)]
                y[i] = data[(i + self.seq_length):(i + self.seq_length + self.forecast_horizon)]
            
            # Log progress for large datasets
            if total_windows > 50000:
                logging.info(f"Processed {batch_end}/{total_windows} windows...")
        
        logging.info(f"Window creation completed. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def create_tf_dataset(self, X, y, is_training=True, cache_file=None):
        """Creates an optimized tf.data.Dataset with file-based caching for maximum performance and GPU balance."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            # Advanced data augmentation for training data FIRST
            dataset = dataset.map(
                lambda x, y: self._augment_data(x, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # File-based caching for GPU balance and VRAM efficiency
            if cache_file:
                dataset = dataset.cache(filename=cache_file)
                logging.info(f"Training dataset using file-based cache: {cache_file}")
            else:
                dataset = dataset.cache()
                logging.info("Training dataset using in-memory cache (fallback)")
            
            # Now shuffle the cached data efficiently
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
        else:
            # For validation/test data, use file-based caching if available
            if cache_file:
                dataset = dataset.cache(filename=cache_file)
                logging.info(f"Validation/Test dataset using file-based cache: {cache_file}")
            else:
                dataset = dataset.cache()
                logging.info("Validation/Test dataset using in-memory cache (fallback)")
        
        # Batch and prefetch operations come after caching
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Add parallel processing for better GPU utilization
        if is_training:
            dataset = dataset.map(
                lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Log dataset optimization status
        if is_training:
            if cache_file:
                logging.info(f"Training dataset created with file-based caching: cache({cache_file}) -> shuffle() -> batch() -> prefetch()")
                logging.info(f"File cache will be created after first epoch for {self.batch_size}x speedup in subsequent epochs")
                logging.info(f"GPU VRAM will be freed for model training and multi-GPU balance")
            else:
                logging.info(f"Training dataset created with in-memory caching: cache() -> shuffle() -> batch() -> prefetch()")
                logging.info(f"Dataset will be cached in memory after first epoch for {self.batch_size}x speedup in subsequent epochs")
        else:
            if cache_file:
                logging.info(f"Validation/Test dataset created with file-based caching for consistent performance")
            else:
                logging.info(f"Validation/Test dataset created with in-memory caching for consistent performance")
        
        return dataset
    
    def _augment_data(self, x, y):
        """
        Advanced data augmentation techniques for time series data.
        - Gaussian noise injection
        - Amplitude scaling
        - REMOVED explicit device placement to allow MirroredStrategy to work correctly
        """
        # Gaussian noise injection (small amount to improve robustness)
        noise_std = 0.01
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_std, dtype=x.dtype)
        x = x + noise
        
        # Amplitude scaling (slight random scaling)
        scale_factor = tf.random.uniform(shape=[], minval=0.95, maxval=1.05, dtype=x.dtype)
        x = x * scale_factor
        
        # Ensure values stay within reasonable bounds
        x = tf.clip_by_value(x, -2.0, 2.0)  # Clip to prevent extreme values
    
        return x, y

# =============================================================================
# 3. OPTIMIZED LSTM MODEL CLASS
# =============================================================================
class OptimizedLSTMForecaster:
    """
    DIRECT MANY-TO-MANY LSTM FORECASTER - Predicts all 96 steps in one forward pass

    KEY ADVANTAGES OVER RECURSIVE APPROACH:
    1. NO recursion: Predicts all 96 steps at once
    2. NO error accumulation: Each prediction is independent
    3. NO cuDNN errors: Compatible with optimized LSTM kernels
    4. 50-100x faster training: 1 forward pass vs 96
    5. Better long-horizon accuracy: Directly optimized for all horizons

    ARCHITECTURE: Encoder-Decoder
    - Input: (batch_size, 96, 1) - 96 historical time steps
    - Output: (batch_size, 96, 1) - ALL 96 future steps predicted at once
    - Encoder: CNN + LSTM to compress input sequence to context vector
    - Decoder: LSTM to generate output sequence from context
    - TimeDistributed Dense: Independent prediction for each output timestep

    TRAINING:
    - Multi-horizon loss: Weighted across all 96 forecast steps
    - Near-term emphasis: Higher weights for early predictions
    - Simple and stable: No scheduled sampling, no teacher forcing complexity

    PRODUCTION USAGE:
    - Single forward pass for 96-step forecast
    - No need for recursive calls
    - Consistent with training methodology
    """
    def __init__(self, seq_length=96, n_features=1, forecast_horizon=96):
        self.seq_length = seq_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.history = None
        self.train_loss_metric = None
        self.val_loss_metric = None

    def build_model(self, strategy):
        """
        DIRECT MANY-TO-MANY ARCHITECTURE: Input 96 steps â†’ Output 96 steps in ONE forward pass

        KEY DIFFERENCES FROM RECURSIVE MODEL:
        - Input: (batch_size, 96, 1) - 96 historical time steps
        - Output: (batch_size, 96, 1) - ALL 96 future steps predicted at once
        - NO recursion, NO error accumulation, NO cuDNN issues
        - Much faster training (1 forward pass vs 96)
        - Direct optimization of all forecast horizons

        ARCHITECTURE: Encoder-Decoder with CNN-LSTM-Attention
        1. Encoder: CNN + LSTM to compress input sequence
        2. Decoder: LSTM to generate output sequence
        3. Attention: Focus on important input timesteps
        4. TimeDistributed Dense: Output layer for all 96 steps
        """
        # --- Hyperparameters ---
        lstm_units_encoder = 256
        lstm_units_decoder = 128
        dropout_rate = 0.3
        learning_rate = 1e-4  # Slightly higher LR for direct forecasting
        cnn_filters = 128

        logging.info("Building DIRECT Many-to-Many CNN-LSTM model...")
        logging.info(f"Architecture: Encoder CNN + LSTM({lstm_units_encoder}) -> Decoder LSTM({lstm_units_decoder}) -> 96 outputs")

        with strategy.scope():
            # Input layer
            inputs = tf.keras.layers.Input(shape=(self.seq_length, self.n_features))

            # ========================
            # ENCODER: Process input sequence
            # ========================

            # CNN layers for local pattern extraction
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

            # MaxPooling to reduce sequence length (96 -> 48)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

            # Additional CNN layer
            x = tf.keras.layers.Conv1D(filters=cnn_filters * 2, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

            # LSTM encoder - compress to context vector
            encoder_output = tf.keras.layers.LSTM(
                lstm_units_encoder,
                return_sequences=False,  # Return only final state
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform'
            )(x)
            encoder_output = tf.keras.layers.Dropout(dropout_rate)(encoder_output)

            # ========================
            # DECODER: Generate output sequence
            # ========================

            # Repeat encoder output for each of 96 output timesteps
            decoder_input = tf.keras.layers.RepeatVector(self.forecast_horizon)(encoder_output)

            # LSTM decoder - generate sequence
            x = tf.keras.layers.LSTM(
                lstm_units_decoder,
                return_sequences=True,  # Return full sequence
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform'
            )(decoder_input)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

            # Optional: Second decoder LSTM layer
            x = tf.keras.layers.LSTM(
                lstm_units_decoder // 2,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform'
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.5)(x)

            # ========================
            # OUTPUT: TimeDistributed Dense for all 96 steps
            # ========================

            # Dense layer applied to each timestep independently
            outputs = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, activation='linear', dtype='float32'),
                name='output_layer'
            )(x)

            # Soft clipping for stability
            # outputs = tf.keras.layers.Lambda(
            #     lambda x: tf.clip_by_value(x, -3.0, 3.0),
            #     name='soft_output_clipping',
            #     dtype='float32'
            # )(outputs)

            outputs = ClipByValue(-3.0, 3.0, name="soft_output_clipping")(outputs)


            # Create the model
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Optimizer
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )

            # Huber loss - robust to outliers
            self.loss_fn = tf.keras.losses.Huber(delta=1.0)

            # Create metrics
            self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

            # Compile the model
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                metrics=[tf.keras.metrics.MeanAbsoluteError()],
                steps_per_execution=10
            )

            # Build optimizer variables
            def build_optimizer_vars():
                dummy_input = tf.zeros((1, self.seq_length, self.n_features), dtype=tf.float32)

                # ----- ISSUE -----
                # dummy forward/backward pass must match whatever output shape that specific model build produces.
                # AND FOLLOWING IS ALSO WRONG FOR SOLAR BECAUSE YOUR TARGET Y IS (N, forecast_horizon, 1), NOT (N, forecast_horizon, n_features).
                # FOLLOWING NEEDS TO BE UPDATED TO WORK WITH NEW MANY-TO-MANY OUTPUT LOGIC:

                # dummy_output = tf.zeros((1, self.forecast_horizon, self.n_features), dtype=tf.float32)  # 96 outputs!

                # ----- ISSUE END -----

                # ----- PATCH START -----

                dummy_output = tf.zeros((1, self.forecast_horizon, 1), dtype=tf.float32)

                # ----- PATCH END -----

                with tf.GradientTape() as tape:
                    dummy_pred = self.model(dummy_input, training=True)
                    dummy_loss = self.loss_fn(dummy_output, dummy_pred)

                trainable_vars = self.model.trainable_variables
                dummy_grads = tape.gradient(dummy_loss, trainable_vars)
                self.optimizer.apply_gradients(zip(dummy_grads, trainable_vars))
                return dummy_loss

            strategy.run(build_optimizer_vars)
            logging.info("Optimizer variables initialized successfully across all GPUs")

        self.model.summary(print_fn=logging.info)
        logging.info(f"Model output shape: (batch_size, {self.forecast_horizon}, 1)")
        return self.model

    def build_tunable_model(self, hp, strategy):
        """
        Builds a tunable SOTA CNN-LSTM with Attention model for Keras Tuner.
        This method is used by the hyperparameter optimization process.
        
        EXPANDED SEARCH SPACE: Leverages available GPU memory (75GB A100) to find
        larger, more powerful architectures that can better handle complex patterns.
        """
        logging.info("Building tunable SOTA CNN-LSTM with Attention model for Keras Tuner...")
        logging.info("Using EXPANDED search space for larger models (leverages 75GB A100 memory)...")
        
        with strategy.scope():
            # EXPANDED Hyperparameter search spaces - Leveraging massive GPU memory headroom
            # Your observation showed only 2.17GB peak usage - we have ~70GB+ available!
            lstm_units_1 = hp.Int('lstm_units_1', min_value=256, max_value=1024, step=256)
            lstm_units_2 = hp.Int('lstm_units_2', min_value=128, max_value=512, step=128)
            lstm_units_3 = hp.Int('lstm_units_3', min_value=64, max_value=256, step=64)
            dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=128)
            dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
            cnn_filters = hp.Int('cnn_filters', min_value=64, max_value=256, step=64)
            attention_heads = hp.Int('attention_heads', min_value=4, max_value=12, step=2)
            
            logging.info(f"EXPANDED Trial hyperparameters: CNN({cnn_filters}) -> LSTM({lstm_units_1}->{lstm_units_2}->{lstm_units_3}) -> Attention({attention_heads}) -> Dense({dense_units}), Dropout({dropout_rate}), LR({learning_rate:.6f})")
            logging.info(f"Memory headroom: ~70GB+ available (current usage: ~2GB peak)")
            
            # Input layer
            inputs = tf.keras.layers.Input(shape=(self.seq_length, self.n_features))
            
            # 1D CNN layers with tunable filters - EXPANDED for more complex patterns
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)
            
            # MaxPooling to reduce sequence length
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            
            # Additional CNN layers for higher-level feature extraction
            x = tf.keras.layers.Conv1D(filters=cnn_filters * 2, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)
            
            # NEW: Additional CNN layer for even more complex pattern recognition
            x = tf.keras.layers.Conv1D(filters=cnn_filters * 2, kernel_size=5, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)
            
            # LSTM layers
            x = tf.keras.layers.LSTM(
                lstm_units_1,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform'
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.LSTM(
                lstm_units_2,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform'
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.LSTM(
                lstm_units_3,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform'
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            
            # Multi-Head Attention with tunable heads
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=attention_heads, key_dim=lstm_units_3 // attention_heads, dropout=dropout_rate * 0.5
            )(x, x)
            
            # Residual connection and normalization
            x = tf.keras.layers.Add()([x, attention_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # Temporal-aware aggregation (preserves temporal information)
            # Use weighted attention for temporal importance
            attention_weights = tf.keras.layers.Dense(1, activation='softmax', name='temporal_attention_tunable')(x)
            weighted_features = tf.keras.layers.Multiply()([x, attention_weights])
            x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='temporal_aggregation_tunable')(weighted_features)

            # Dense layers - EXPANDED for more complex feature transformation
            x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.5)(x)
            
            # NEW: Additional dense layer for intermediate feature processing
            x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.4)(x)
            
            x = tf.keras.layers.Dense(dense_units // 2, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate * 0.3)(x)

            # -------------------------------------------------------------------------------------------------
            # OLD LOGIC: MANY-TO-ONE, CAUSED ISSUES WITH SOLAR DATA
            # ISSUES WERE: ValueError: Found input variables with inconsistent numbers of samples: [983040, 10240]
            # EARLIER STEPS IN THE CODE WORK ON THE MANY-TO-MANY LOGIC SO WE NEED AN UPDATED OUTPUT LAYER
            # THAT WORKS IN A MANY TO MANY WAY
            
            # # Output layer - CRITICAL: Single output for many-to-one architecture
            # # Linear activation with soft clipping for stability
            # outputs = tf.keras.layers.Dense(1, activation='linear', dtype='float32')(x)
            # outputs = tf.keras.layers.Lambda(
            #     lambda x: tf.clip_by_value(x, -3.0, 3.0),
            #     name='soft_output_clipping_tunable',
            #     dtype='float32'
            # )(outputs)

            # -------------------------------------------------------------------------------------------------

            # ------ PATCH FOR ABOVE ------

            # Output layer - Multi-step horizon output (matches y shape: (N, forecast_horizon, 1))
            # Linear activation with soft clipping for stability
            outputs = tf.keras.layers.Dense(self.forecast_horizon, activation='linear', dtype='float32')(x)
            # outputs = tf.keras.layers.Lambda(
            #     lambda x: tf.clip_by_value(x, -3.0, 3.0),
            #     name='soft_output_clipping_tunable',
            #     dtype='float32'
            # )(outputs)

            outputs = ClipByValue(-3.0, 3.0, name="soft_output_clipping_tunable")(outputs)

            outputs = tf.keras.layers.Reshape((self.forecast_horizon, 1), name='forecast_horizon_reshape')(outputs)

            # ------ END PATCH ------

            # Create the model
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Create optimizer and loss function
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            # Huber loss - robust to outliers, stable gradients
            self.loss_fn = tf.keras.losses.Huber(delta=1.0)
            
            # Create metrics inside the strategy scope
            # This makes them "strategy-aware" and allows multi-GPU training to work
            self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
            
            # Compile the model
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                metrics=[tf.keras.metrics.MeanAbsoluteError()],
                steps_per_execution=10
            )
            
            # Build optimizer variables
            def build_optimizer_vars():
                dummy_input = tf.zeros((1, self.seq_length, self.n_features), dtype=tf.float32)

                # ----- ISSUE -----
                # dummy forward/backward pass must match whatever output shape that specific model build produces.
                # FOLLOWING NEEDS TO BE UPDATED TO WORK WITH NEW MANY-TO-MANY OUTPUT LOGIC
                # dummy_output = tf.zeros((1, 1), dtype=tf.float32)
                # ----- ISSUE END-----

                # ----- PATCH START -----
                
                dummy_output = tf.zeros((1, self.forecast_horizon, 1), dtype=tf.float32)
                
                # ----- PATCH END -----
                
                with tf.GradientTape() as tape:
                    dummy_pred = self.model(dummy_input, training=True)
                    dummy_loss = self.loss_fn(dummy_output, dummy_pred)
                
                trainable_vars = self.model.trainable_variables
                dummy_grads = tape.gradient(dummy_loss, trainable_vars)
                self.optimizer.apply_gradients(zip(dummy_grads, trainable_vars))
                return dummy_loss
            
            strategy.run(build_optimizer_vars)
            logging.info("Tunable model built successfully across all GPUs")
            
            # Log expected model complexity for this trial
            total_params = self.model.count_params()
            logging.info(f"Model complexity: {total_params:,} total parameters")
            logging.info(f"Expected GPU memory usage: ~{total_params * 4 / 1024**3:.2f} GB (float32)")
            logging.info(f"Memory headroom: ~{75 - total_params * 4 / 1024**3:.2f} GB available")
        
        return self.model

    @tf.function
    def _train_step(self, x_batch, y_batch):
        """
        SIMPLIFIED DIRECT TRAINING STEP - No recursion, no while_loop!

        - Model predicts all 96 steps in ONE forward pass
        - Loss computed across all horizons simultaneously
        - Optional: Weighted loss to emphasize near-term accuracy
        - cuDNN compatible, 50-100x faster than recursive approach
        """
        with tf.GradientTape() as tape:
            # Single forward pass predicts all 96 steps
            predictions = self.model(x_batch, training=True)  # Shape: (batch, 96, 1)

            # Multi-horizon loss with optional decay weights
            # Emphasize near-term predictions (higher weight for early timesteps)
            horizon_weights = tf.exp(-tf.range(self.forecast_horizon, dtype=tf.float32) / self.forecast_horizon)
            horizon_weights = horizon_weights / tf.reduce_sum(horizon_weights)  # Normalize

            # Compute per-timestep loss
            per_step_loss = self.loss_fn(y_batch, predictions)

            # Weighted average across all forecast steps
            loss = tf.reduce_mean(horizon_weights * per_step_loss)

        # Standard gradient descent
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)

    @tf.function
    def _test_step(self, x_batch, y_batch):
        """
        SIMPLIFIED VALIDATION STEP - Direct prediction, no recursion!

        - Model predicts all 96 steps in one forward pass
        - Same as production usage (no teacher forcing needed)
        - Consistent with training approach
        - Fast and cuDNN compatible
        """
        # Single forward pass for all 96 predictions
        predictions = self.model(x_batch, training=False)  # Shape: (batch, 96, 1)

        # Compute loss across all horizons
        # Use same weighted loss as training for consistency
        horizon_weights = tf.exp(-tf.range(self.forecast_horizon, dtype=tf.float32) / self.forecast_horizon)
        horizon_weights = horizon_weights / tf.reduce_sum(horizon_weights)

        per_step_loss = self.loss_fn(y_batch, predictions)
        loss = tf.reduce_mean(horizon_weights * per_step_loss)

        self.val_loss_metric.update_state(loss)

    def fit(self, train_dataset, validation_dataset, epochs=100, patience=15):
        """
        SIMPLIFIED DIRECT TRAINING - No scheduled sampling, no teacher forcing!

        - Each epoch does one pass through training data
        - Model predicts all 96 steps directly
        - Much faster and more stable than recursive approach
        - Early stopping based on validation loss
        """
        logging.info(f"Starting DIRECT many-to-many training for {epochs} epochs...")
        logging.info("Key difference: Predicting all 96 steps in one forward pass (no recursion)")
        self.history = {'loss': [], 'val_loss': []}

        best_val_loss = float('inf')
        patience_counter = 0

        lr_reduction_patience = 10
        lr_reduction_counter = 0
        min_lr = 1e-7

        # checkpoint_dir = os.path.join(os.getcwd(), 'logs', 'model_checkpoints')
        checkpoint_dir = os.path.join(os.getcwd(), 'model_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
        logging.info(f"Model checkpointing enabled. Best model will be saved to {best_model_path}")

        for epoch in range(epochs):
            logging.info(f"\nEpoch {epoch+1}/{epochs}")

            # Learning rate scheduling can remain as is.
            if epoch == 0:
                warmup_lr = self.optimizer.learning_rate.numpy() * 0.1
                self.optimizer.learning_rate.assign(warmup_lr)
                logging.info(f"Warmup phase: Setting learning rate to {warmup_lr:.6f}")
            elif epoch == 5:
                full_lr = self.optimizer.learning_rate.numpy() * 10
                self.optimizer.learning_rate.assign(full_lr)
                logging.info(f"Warmup complete: Restoring learning rate to {full_lr:.6f}")
            elif epoch > 20:
                progress = (epoch - 20) / (epochs - 20)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                new_lr = max(self.optimizer.learning_rate.numpy() * cosine_decay, min_lr)
                self.optimizer.learning_rate.assign(new_lr)

            self.train_loss_metric.reset_state()
            self.val_loss_metric.reset_state()

            train_steps = 0
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                # Simple direct training - no sampling_prob needed!
                self._train_step(x_batch, y_batch)
                train_steps += 1
                if step % 100 == 0:
                    logging.info(f"Epoch {epoch+1}, Step {step}, Training in progress...")
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
            logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.6f} (Completed {train_steps} steps)")

            val_steps = 0
            for x_batch_val, y_batch_val in validation_dataset:
                self._test_step(x_batch_val, y_batch_val)
                val_steps += 1
            
            val_loss = self.val_loss_metric.result().numpy()
            self.history['val_loss'].append(val_loss)
            logging.info(f"Epoch {epoch+1} Val Loss: {val_loss:.6f} (Completed {val_steps} steps)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                lr_reduction_counter = 0
                try:
                    self.model.save_weights(best_model_path)
                    logging.info(f"New best validation loss: {best_val_loss:.6f} - Model saved to {best_model_path}")
                except Exception as e:
                    logging.warning(f"Failed to save model weights: {e}")
            else:
                patience_counter += 1
                lr_reduction_counter += 1
                logging.info(f"No improvement for {patience_counter} epochs")
                if lr_reduction_counter >= lr_reduction_patience:
                    current_lr = self.optimizer.learning_rate.numpy()
                    if current_lr > min_lr:
                        new_lr = current_lr * 0.5
                        self.optimizer.learning_rate.assign(new_lr)
                        logging.info(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f} due to plateau")
                        lr_reduction_counter = 0

            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                try:
                    self.model.load_weights(best_model_path)
                    logging.info(f"Best model weights restored from {best_model_path}")
                except Exception as e:
                    logging.warning(f"Failed to load best weights: {e}")
                break
        
        return self.model

    def evaluate_model(self, test_dataset):
        """
        DIRECT EVALUATION - No recursion!

        - Model predicts all 96 steps in one forward pass
        - No error accumulation from recursive predictions
        - Consistent with how model was trained
        - Returns per-step metrics for analyzing forecast horizon performance
        """
        logging.info("Evaluating model on test set using DIRECT forecasting...")
        logging.info("Model predicts all 96 steps in one forward pass (no recursion)")

        all_forecasts = []
        all_actuals = []

        for x_batch, y_batch in test_dataset:
            # Single forward pass predicts all 96 steps
            predictions = self.model(x_batch, training=False)  # Shape: (batch_size, 96, 1)

            all_forecasts.append(predictions.numpy())
            all_actuals.append(y_batch.numpy())

        # Concatenate all batches
        y_pred_full = np.concatenate(all_forecasts, axis=0)
        y_true_full = np.concatenate(all_actuals, axis=0)

        # Flatten for overall metrics
        y_pred_flat = y_pred_full.flatten()
        y_true_flat = y_true_full.flatten()

        # Compute metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + epsilon))) * 100

        # Convert numpy types to native Python types for JSON serialization
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape)
        }

        logging.info("="*60)
        logging.info("DIRECT FORECASTING PERFORMANCE (All 96 steps predicted at once)")
        logging.info("="*60)
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # Calculate MAE for each forecast step (horizon analysis)
        per_step_mae = np.mean(np.abs(y_true_full - y_pred_full), axis=(0, 2))

        logging.info("\nPer-Horizon Performance:")
        logging.info(f"  Steps 1-5 MAE:   {[f'{x:.4f}' for x in per_step_mae[:5]]}")
        logging.info(f"  Steps 46-50 MAE: {[f'{x:.4f}' for x in per_step_mae[45:50]]}")
        logging.info(f"  Steps 92-96 MAE: {[f'{x:.4f}' for x in per_step_mae[-5:]]}")

        # Analyze error accumulation pattern
        early_mae = np.mean(per_step_mae[:24])  # First 6 hours
        mid_mae = np.mean(per_step_mae[24:72])  # Middle 12 hours
        late_mae = np.mean(per_step_mae[72:])   # Last 6 hours

        logging.info(f"\nError by Forecast Period:")
        logging.info(f"  Early (0-6h):   {early_mae:.4f}")
        logging.info(f"  Middle (6-18h): {mid_mae:.4f}")
        logging.info(f"  Late (18-24h):  {late_mae:.4f}")
        logging.info("="*60)

        # Return flat predictions for plots AND per-step metrics
        return metrics, y_pred_flat, y_true_flat, per_step_mae


# =============================================================================
# 4. KERAS TUNER INTEGRATION CLASS
# =============================================================================
class TimeSeriesHyperModel(keras_tuner.HyperModel):
    """
    Keras Tuner HyperModel for time series forecasting.
    This class enables automatic hyperparameter optimization.
    """
    def __init__(self, seq_length=96, n_features=1, forecast_horizon=96, strategy=None):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.strategy = strategy
        self.forecaster = None

    def build(self, hp):
        """Builds the model with hyperparameters from Keras Tuner."""
        self.forecaster = OptimizedLSTMForecaster(
            seq_length=self.seq_length,
            n_features=self.n_features,
            forecast_horizon=self.forecast_horizon
        )
        
        # Build the tunable model
        self.forecaster.build_tunable_model(hp, self.strategy)
        return self.forecaster.model

    def fit(self, hp, model, train_dataset, **kwargs):
        """Custom fit method compatible with Keras Tuner's calling convention."""
        # Keras Tuner passes validation data as 'validation_data' kwarg
        validation_dataset = kwargs.get('validation_data')
        if validation_dataset is None:
            raise ValueError("validation_data must be provided to tuner.search(..., validation_data=...) to use custom fit")
        
        # Training config
        epochs = kwargs.get('epochs', 50)
        patience = kwargs.get('patience', 10)
        
        # Use the custom training loop
        self.forecaster.fit(train_dataset, validation_dataset, epochs=epochs, patience=patience)
        
        # Return a dict with the objective metric so Keras Tuner can read it
        best_val_loss = float(min(self.forecaster.history['val_loss']))
        return {"val_loss": best_val_loss}

# =============================================================================
# 5. VISUALIZATION CLASS
# =============================================================================
class TimeSeriesVisualizer:
    """Handles all plotting and visualization tasks."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        plt.style.use('default')

    def plot_data_overview(self, df):
        """Creates comprehensive data overview plots and saves them."""
        fig = df['value_filled'].plot(figsize=(15, 5), title='Time Series Overview (v3 Preprocessed)').get_figure()
        plt.savefig(os.path.join(self.output_dir, 'data_overview.png'))
        plt.close(fig)

    def plot_training_history(self, history):
        """Plots the training and validation loss and saves the figure."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss Over Epochs (MAE Loss)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MAE)')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close(fig)

    def plot_predictions(self, y_true, y_pred, n_samples=500):
        """Plots comprehensive predictions vs actual values with advanced analysis."""
        # Create subplots for different views
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Advanced Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series comparison
        axes[0, 0].plot(y_true[:n_samples], label='Actual', linewidth=1.5, color='blue')
        axes[0, 0].plot(y_pred[:n_samples], label='Predicted', linewidth=1.5, alpha=0.8, color='red')
        axes[0, 0].set_title(f'Time Series Comparison (First {n_samples} Samples)')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot with perfect prediction line
        axes[0, 1].scatter(y_true[:n_samples], y_pred[:n_samples], alpha=0.6, s=20)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_title('Predicted vs Actual Values')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals analysis
        residuals = y_true[:n_samples] - y_pred[:n_samples]
        axes[1, 0].plot(residuals, color='green', linewidth=1)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals histogram
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual Values')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'advanced_predictions_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save detailed analysis data
        analysis_data = {
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals)),
            'residuals_mae': float(np.mean(np.abs(residuals))),
            'correlation': float(np.corrcoef(y_true[:n_samples], y_pred[:n_samples])[0, 1])
        }
        
        analysis_path = os.path.join(self.output_dir, 'prediction_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=4)
        
        logging.info(f"Advanced prediction analysis saved to {analysis_path}")
        logging.info(f"Residuals - Mean: {analysis_data['residuals_mean']:.4f}, Std: {analysis_data['residuals_std']:.4f}")
        logging.info(f"Correlation: {analysis_data['correlation']:.4f}")

    def plot_error_by_horizon(self, per_step_mae):
        """Plots the MAE for each step of the forecast horizon."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, len(per_step_mae) + 1), per_step_mae, marker='o', linestyle='-', color='purple')
        ax.set_title('Mean Absolute Error (MAE) vs. Forecast Horizon')
        ax.set_xlabel('Forecast Step (Horizon)')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_xticks(range(0, len(per_step_mae) + 1, 12)) # Ticks every 12 steps
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_by_horizon.png'), dpi=300)
        plt.close(fig)
        logging.info("Error by horizon plot saved.")

# =============================================================================
# 6. HYPERPARAMETER OPTIMIZATION FUNCTION
# =============================================================================
def run_hyperparameter_optimization(train_dataset, validation_dataset, strategy, output_dir, max_trials=40, epochs_per_trial=30):
    """
    Runs hyperparameter optimization using Keras Tuner.
    
    EXPANDED SEARCH SPACE: Leverages available GPU memory (75GB A100) to find
    larger, more powerful architectures that can better handle complex patterns.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        strategy: TensorFlow distribution strategy
        output_dir: Directory to save results
        max_trials: Maximum number of trials to run (increased default)
        epochs_per_trial: Number of epochs per trial (increased for better convergence)
    """
    logging.info("\n" + "="*50)
    logging.info("EXPANDED HYPERPARAMETER OPTIMIZATION WITH KERAS TUNER RANDOMSEARCH")
    logging.info("="*50)
    logging.info(f"Max trials: {max_trials} (increased for thorough search)")
    logging.info(f"Epochs per trial: {epochs_per_trial} (increased for better convergence)")
    logging.info("EXPANDED search space: LSTM(256-1024), CNN(64-256), Dense(128-512)")
    logging.info("Memory headroom: ~70GB+ available (current usage: ~2GB peak)")
    logging.info("CRITICAL FIX: Using RandomSearch instead of Hyperband for reliability")
    logging.info("="*50)
    
    # Clean up any existing keras_tuner directory to avoid checkpoint conflicts
    keras_tuner_dir = os.path.join(output_dir, 'keras_tuner')
    if os.path.exists(keras_tuner_dir):
        import shutil
        logging.info(f"Removing existing keras_tuner directory: {keras_tuner_dir}")
        shutil.rmtree(keras_tuner_dir)
    
    # Create hypermodel
    hypermodel = TimeSeriesHyperModel(
        seq_length=96,
        n_features=1,
        forecast_horizon=96,
        strategy=strategy
    )
    
    # RandomSearch eliminates the FileNotFoundError that plagued Hyperband
    tuner = keras_tuner.RandomSearch(
        hypermodel,
        objective=keras_tuner.Objective('val_loss', direction='min'),
        max_trials=max_trials,
        executions_per_trial=1,  # Single execution per trial for efficiency
        directory=os.path.join(output_dir, 'keras_tuner'),
        project_name='timeseries_forecasting',
        overwrite=True,  # Force overwrite to avoid checkpoint conflicts
        seed=42  # Reproducible results
    )
    
    # Run the search with error handling
    logging.info("Starting RandomSearch hyperparameter search...")
    try:
        tuner.search(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs_per_trial,
            verbose=1
        )
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters(1)[0]
        logging.info("\nBest hyperparameters found:")
        for param, value in best_hp.values.items():
            logging.info(f"{param}: {value}")
        
        # Save best hyperparameters
        best_hp_path = os.path.join(output_dir, 'best_hyperparameters.json')
        with open(best_hp_path, 'w') as f:
            json.dump(best_hp.values, f, indent=4)
        logging.info(f"Best hyperparameters saved to {best_hp_path}")
        
        # Build and return the best model
        best_model = tuner.hypermodel.build(best_hp)
        return best_model, best_hp
        
    except Exception as e:
        logging.error(f"RandomSearch hyperparameter optimization failed: {e}")
        logging.info("Falling back to default hyperparameters...")
        
        # Return a model with default hyperparameters
        default_forecaster = OptimizedLSTMForecaster(
            seq_length=96,
            n_features=1,
            forecast_horizon=96
        )
        default_model = default_forecaster.build_model(strategy)
        
        # Create default hyperparameters dict - MEMORY OPTIMIZED
        default_hp = {
            'lstm_units_1': 384,
            'lstm_units_2': 192,
            'lstm_units_3': 632,
            'dense_units': 256,
            'dropout_rate': 0.3,
            'learning_rate': 4.7688800201954735e-05
        }
        
        # Save default hyperparameters
        default_hp_path = os.path.join(output_dir, 'default_hyperparameters.json')
        with open(default_hp_path, 'w') as f:
            json.dump(default_hp, f, indent=4)
        logging.info(f"Default hyperparameters saved to {default_hp_path}")
        
        return default_model, default_hp

# =============================================================================
# 7. MAIN EXECUTION FUNCTION
# =============================================================================
def main(args):
    """Main execution function with the complete, corrected pipeline."""
    # Setup logging to save to the output directory
    setup_logging(args.logs_dir)
    
    # Log SLURM environment information
    if 'SLURM_JOB_ID' in os.environ:
        logging.info("="*60)
        logging.info("SLURM ENVIRONMENT DETECTED")
        logging.info("="*60)
        logging.info(f"Job ID: {os.environ['SLURM_JOB_ID']}")
        logging.info(f"Job Name: {os.environ.get('SLURM_JOB_NAME', 'Not Set')}")
        logging.info(f"Node Count: {os.environ.get('SLURM_NNODES', 'Not Set')}")
        logging.info(f"GPUs per Node: {os.environ.get('SLURM_GPUS_PER_NODE', 'Not Set')}")
        logging.info(f"CPUs per Task: {os.environ.get('SLURM_CPUS_PER_TASK', 'Not Set')}")
        logging.info(f"Partition: {os.environ.get('SLURM_PARTITION', 'Not Set')}")
        logging.info(f"QOS: {os.environ.get('SLURM_QOS', 'Not Set')}")
        logging.info("="*60)
    
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("cpu_only enabled: CUDA_VISIBLE_DEVICES=-1 (forcing CPU-only)")

    if not args.cpu_only:
        configure_gpu()
    else:
        logging.info("Skipping GPU configuration because cpu_only=True")

    # --- Multi-GPU Strategy Setup with SLURM Integration ---
    # Check if we're in a SLURM environment and respect existing settings
    if 'SLURM_JOB_ID' in os.environ:
        logging.info(f"Running in SLURM environment: Job ID {os.environ['SLURM_JOB_ID']}")
        logging.info(f"SLURM_GPUS_ON_NODE: {os.environ.get('SLURM_GPUS_ON_NODE', 'Not Set')}")
        logging.info(f"SLURM_GPUS_PER_NODE: {os.environ.get('SLURM_GPUS_PER_NODE', 'Not Set')}")
        
        # Log any NCCL environment variables that are set by SLURM script
        nccl_vars = ['NCCL_DEBUG', 'NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE', 
                     'NCCL_SOCKET_IFNAME', 'NCCL_ASYNC_ERROR_HANDLING']
        logging.info("NCCL Environment Variables:")
        for var in nccl_vars:
            value = os.environ.get(var, 'Not Set')
            if value != 'Not Set':
                logging.info(f"  {var}: {value} (set by SLURM script)")
            else:
                logging.info(f"  {var}: {value} (using default)")
        
        # Note: For modern HPC clusters like ASU Sol, default NCCL settings are optimal
        # Only set these variables in SLURM scripts if troubleshooting is needed
    else:
        logging.info("Running outside SLURM environment")
        logging.info("Using default NCCL settings (recommended for modern systems)")
    
    # try:
    #     strategy = tf.distribute.MirroredStrategy()
    #     logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")
    #     logging.info("Multi-GPU strategy initialized successfully")
    # except Exception as e:
    #     logging.warning(f"Failed to initialize multi-GPU strategy: {e}")
    #     logging.info("Falling back to single GPU strategy")
    #     strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    #     logging.info("Single GPU strategy initialized")

    if args.cpu_only:
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        logging.info("CPU-only strategy initialized: OneDeviceStrategy(/CPU:0)")
    else:
        try:
            strategy = tf.distribute.MirroredStrategy()
            logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")
            logging.info("Multi-GPU strategy initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize multi-GPU strategy: {e}")
            logging.info("Falling back to single GPU strategy")
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            logging.info("Single GPU strategy initialized")


    
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    logging.info(f"Global batch size: {global_batch_size}")
    logging.info(f"Per-GPU batch size: {args.batch_size}")
    logging.info(f"Strategy type: {type(strategy).__name__}")
    
    # CRITICAL: Verify multi-GPU setup is working
    if strategy.num_replicas_in_sync > 1:
        logging.info("âœ… MULTI-GPU SETUP VERIFIED: Training will use multiple GPUs")
        logging.info(f"   - Devices: {strategy.extended.worker_devices}")
        logging.info(f"   - Replicas: {strategy.num_replicas_in_sync}")
        logging.info(f"   - Expected GPU utilization: Both GPUs should show activity")
    else:
        logging.warning("âš ï¸  SINGLE-GPU MODE: Only one GPU will be used")
        logging.warning("   - Check for strategy.scope() errors in the logs")
        logging.warning("   - Verify CUDA_VISIBLE_DEVICES is set correctly")
    
    data_processor = TimeSeriesDataProcessor(
        seq_length=96,
        forecast_horizon=96, # Target y is 96 steps for loss calculation
        batch_size=global_batch_size,
        dataset_kind=args.dataset_kind
    )

    visualizer = TimeSeriesVisualizer(args.output_dir)

    logging.info("\n" + "="*50)
    logging.info("STEP 1: DATA LOADING AND PREPROCESSING (v3 + Daily Differencing)")
    logging.info("="*50)
    logging.info("v3 Preprocessing Features:")
    logging.info("- Gap classification: Short (<7h) vs Long (â‰¥7h)")
    logging.info("- Short gaps: Hierarchical fill (D-1 â†’ D-7 â†’ ToD median â†’ LOCF)")
    logging.info("- Long gaps: Time-of-Day median for stability")
    logging.info("- Converts zeros to NaN for proper gap detection")
    logging.info("NEW: Daily Seasonal Differencing (lag=96):")
    logging.info("- Removes strong daily patterns by predicting changes")
    logging.info("- Makes forecasting problem easier and more accurate")
    logging.info("- Model learns subtle variations rather than entire daily shape")
    logging.info("="*50)
    
    df = data_processor.load_and_process_data(args.data_dir)
    log_memory_usage("after data loading")
    
    visualizer.plot_data_overview(df)
    
    # Save preprocessed series for audit
    preprocessed_path = os.path.join(args.output_dir, 'preprocessed_value_filled.csv')
    df.to_csv(preprocessed_path, index_label='time')
    logging.info(f"Preprocessed data saved to {preprocessed_path}")
    
    data_values = df['value_filled'].values
    
    # Clear dataframe to free memory
    del df
    gc.collect()
    log_memory_usage("after clearing dataframe")
    
    # Handle any remaining NaN values (should be minimal after v3 preprocessing)
    if np.isnan(data_values).any():
        nan_count = np.isnan(data_values).sum()
        logging.warning(f"Found {nan_count} remaining NaN values after v3 preprocessing. Forward-filling...")
        data_values = pd.Series(data_values).fillna(method='ffill').fillna(method='bfill').values
        logging.info("Forward/backward fill completed.")
    
    # =============================================================================
    # NEW: APPLY DAILY SEASONAL DIFFERENCING (LAG=96)
    # =============================================================================
    logging.info("\n" + "="*50)
    logging.info("APPLYING DAILY SEASONAL DIFFERENCING (LAG=96)")
    logging.info("="*50)
    logging.info("This transformation removes strong daily patterns by predicting changes")
    logging.info("rather than absolute values, making the forecasting problem easier.")
    logging.info("="*50)
    
    # Convert to pandas Series for differencing operation
    data_series = pd.Series(data_values)
    
    # Apply daily differencing (lag=96 for 15-minute data: 24h * 4 steps per hour)
    daily_lag = 96
    logging.info(f"Applying daily differencing with lag={daily_lag} (24h * 4 steps per hour)")
    logging.info(f"Original data shape: {len(data_series)}")
    
    # Calculate the differenced values
    differenced_values = data_series.diff(daily_lag)
    
    # Log differencing statistics
    logging.info(f"Differencing statistics:")
    logging.info(f"  - Original range: [{data_series.min():.2f}, {data_series.max():.2f}]")
    logging.info(f"  - Differenced range: [{differenced_values.min():.2f}, {differenced_values.max():.2f}]")
    logging.info(f"  - Original std: {data_series.std():.2f}")
    logging.info(f"  - Differenced std: {differenced_values.std():.2f}")
    
    # The first 96 values will be NaN after differencing, so we must drop them
    original_length = len(differenced_values)
    differenced_values = differenced_values.dropna()
    dropped_count = original_length - len(differenced_values)
    logging.info(f"Dropped {dropped_count} NaN values from differencing (first {daily_lag} steps)")
    logging.info(f"Final differenced data shape: {len(differenced_values)}")
    
    # Convert back to numpy array
    data_values = differenced_values.values
    
    # CRITICAL FIX: Split data BEFORE scaling to prevent data leakage
    train_ratio, val_ratio = 0.8, 0.1
    train_idx = int(len(data_values) * train_ratio)
    val_idx = int(len(data_values) * (train_ratio + val_ratio))
    
    train_data, val_data, test_data = data_values[:train_idx], data_values[train_idx:val_idx], data_values[val_idx:]
    
    logging.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    logging.info(f"Train data range: [{train_data.min():.2f}, {train_data.max():.2f}]")
    logging.info(f"Val data range: [{val_data.min():.2f}, {val_data.max():.2f}]")
    logging.info(f"Test data range: [{test_data.min():.2f}, {test_data.max():.2f}]")

# =============================================================================
# CRITICAL: NORMALIZATION AND DIFFERENCING SETUP FOR PRODUCTION SYSTEMS
# =============================================================================
# This normalization AND differencing process MUST be replicated exactly in all 
# downstream systems (implementation script, HMG4) using the exported parameters.
#
# NORMALIZATION FORMULA:
#   Input:  normalized = (value - min) / (max - min) * 2 - 1
#   Output: denormalized = (predicted + 1) / 2 * (max - min) + min
#
# DIFFERENCING FORMULA (Daily Seasonal):
#   Input:  differenced = value[t] - value[t-96]  (96 = 24h * 4 steps per hour)
#   Output: reconstructed = predicted_change + value[t-96]
#
# WHERE min and max are calculated from training data only (prevents data leakage)
# =============================================================================
    
    # CRITICAL FIX: Fit scaler ONLY on training data to prevent data leakage
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    val_scaled = scaler.transform(val_data.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.reshape(-1, 1))
    
    logging.info("Data leakage prevention: Scaler fitted ONLY on training data")
    logging.info(f"Scaler fitted on range: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")
    logging.info("CRITICAL: These min/max values will be exported for production systems")

    X_train, y_train = data_processor.create_windows(train_scaled)
    log_memory_usage("after creating train windows")
    
    X_val, y_val = data_processor.create_windows(val_scaled)
    log_memory_usage("after creating val windows")
    
    X_test, y_test = data_processor.create_windows(test_scaled)
    log_memory_usage("after creating test windows")
    
    logging.info(f"Train windows: {X_train.shape}, Val windows: {X_val.shape}, Test windows: {X_test.shape}")
    
    # Clear scaled data to free memory
    del train_scaled, val_scaled, test_scaled
    gc.collect()
    log_memory_usage("after clearing scaled data")

    # CRITICAL FIX: Implement file-based caching for GPU balance and VRAM efficiency
    # This ensures both GPUs work efficiently and frees VRAM for model training
    train_cache_file = os.path.join(args.output_dir, 'train_cache')
    val_cache_file = os.path.join(args.output_dir, 'val_cache')
    test_cache_file = os.path.join(args.output_dir, 'test_cache')
    
    logging.info("Implementing file-based caching for optimal GPU performance...")
    logging.info(f"Train cache: {train_cache_file}")
    logging.info(f"Val cache: {val_cache_file}")
    logging.info(f"Test cache: {test_cache_file}")
    
    train_dataset = data_processor.create_tf_dataset(X_train, y_train, cache_file=train_cache_file)
    val_dataset = data_processor.create_tf_dataset(X_val, y_val, is_training=False, cache_file=val_cache_file)
    test_dataset = data_processor.create_tf_dataset(X_test, y_test, is_training=False, cache_file=test_cache_file)
    
    if args.use_tuner:
        logging.info("\n" + "="*50)
        logging.info("STEP 2: EXPANDED HYPERPARAMETER OPTIMIZATION WITH KERAS TUNER RANDOMSEARCH")
        logging.info("="*50)
        logging.info("Running EXPANDED automatic hyperparameter search with RandomSearch...")
        logging.info("NEW: Leveraging ~70GB+ GPU memory headroom for larger models")
        logging.info("Search space: LSTM(256-1024), CNN(64-256), Dense(128-512)")
        logging.info("CRITICAL FIX: RandomSearch eliminates FileNotFoundError for reliable execution")
        logging.info("This should find more powerful architectures to address underfitting")
        logging.info("="*50)
        
        # Run hyperparameter optimization
        best_model, best_hp = run_hyperparameter_optimization(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            strategy=strategy,
            output_dir=args.output_dir,
            max_trials=args.max_trials,
            epochs_per_trial=args.epochs_per_trial
        )
        
        # Create a forecaster with the best model for evaluation
        model_forecaster = OptimizedLSTMForecaster(
            seq_length=96,
            n_features=1,
            forecast_horizon=96
        )
        model_forecaster.model = best_model
        
        # Export the best model in SavedModel format (Keras 3)
        model_path = os.path.join(args.output_dir, 'best_tuned_model_savedmodel')
        best_model.export(model_path)
        logging.info(f"Best tuned model exported (SavedModel) to directory: {model_path}")
        
    else:
        logging.info("\n" + "="*50)
        logging.info("STEP 2: DIRECT MANY-TO-MANY MODEL BUILDING AND TRAINING")
        logging.info("="*50)
        logging.info("DIRECT Encoder-Decoder Architecture: CNN-LSTM")
        logging.info("- Input: (batch_size, 96, 1) - 96 historical time steps")
        logging.info("- Output: (batch_size, 96, 1) - ALL 96 future steps at once")
        logging.info("- KEY: NO recursion, NO error accumulation, NO cuDNN errors")
        logging.info("- Model predicts CHANGES, not absolute values (daily differencing)")
        logging.info("- Training: Direct multi-horizon optimization with weighted loss")
        logging.info("- Evaluation: Single forward pass (consistent with training)")
        logging.info("- Architecture: CNN Encoder -> LSTM Encoder -> LSTM Decoder -> TimeDistributed Dense")
        logging.info("- Training Features: Multi-horizon loss, LR scheduling, Early stopping")
        logging.info("- Loss: Huber loss with horizon decay weights (emphasizes near-term)")
        logging.info("- HPC Optimizations: Multi-GPU (A100 80GB), Mixed precision, XLA compilation")
        logging.info("- PERFORMANCE: 50-100x faster training than recursive approach")
        logging.info("="*50)
        
        model_forecaster = OptimizedLSTMForecaster(
            seq_length=96,
            n_features=1, # Univariate model
            forecast_horizon=96
        )
        
        model_forecaster.build_model(strategy)
        
        start_time = time.time()
        model_forecaster.fit(train_dataset, val_dataset, epochs=args.epochs, patience=args.patience)
        training_time = time.time() - start_time
        logging.info(f"\nTraining completed in {training_time:.2f} seconds")
        
        visualizer.plot_training_history(model_forecaster.history)

        # Export the final model in SavedModel format (Keras 3)
        model_path = os.path.join(args.output_dir, 'final_model_savedmodel')
        model_forecaster.model.export(model_path)
        logging.info(f"Model exported (SavedModel) to directory: {model_path}")

        # -------------------------------
        # CPU-compatible export (optional)
        # -------------------------------
        if args.export_cpu:
            cpu_export_dir = os.path.join(args.output_dir, "final_model_savedmodel_cpu")
            keras_backup_path = os.path.join(args.output_dir, "final_model_for_cpu_export.keras")

            model_forecaster.model.save(keras_backup_path)
            logging.info(f"Saved portable Keras model for CPU export at: {keras_backup_path}")

            script_path = os.path.join(os.path.dirname(__file__), "export_cpu_savedmodel.py")
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"CPU export script not found at: {script_path}")

            cmd = [
                sys.executable,
                script_path,
                "--keras_path", keras_backup_path,
                "--export_dir", cpu_export_dir,
                "--seq_len", str(args.seq_len),
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU-only trace
            env["TF_ENABLE_ONEDNN_OPTS"] = "0"   # optional stability
            env["TF_USE_CUDNN"] = "0"            # extra safety

            logging.info(f"Launching CPU export subprocess: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, env=env)
            logging.info(f"CPU-compatible SavedModel exported to: {cpu_export_dir}")
        else:
            logging.info("Skipping CPU export (run with --export_cpu to enable).")


    logging.info("\n" + "="*50)
    logging.info("STEP 3: MODEL EVALUATION")
    logging.info("="*50)
    
    metrics, y_pred_scaled, y_true_scaled, per_step_mae = model_forecaster.evaluate_model(test_dataset)
    
    # Inverse transform for visualization and final metrics
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    visualizer.plot_predictions(y_true, y_pred)
    
    # NEW: Plot the error by horizon analysis
    visualizer.plot_error_by_horizon(per_step_mae)
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(args.output_dir, 'performance_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Performance metrics saved to {metrics_path}")

    # =============================================================================
    # STEP 4: EXPORTING NORMALIZATION PARAMETERS FOR PRODUCTION SYSTEMS
    # =============================================================================
    logging.info("\n" + "="*50)
    logging.info("STEP 4: EXPORTING NORMALIZATION PARAMETERS FOR PRODUCTION")
    logging.info("="*50)
    
    # Extract the min and max values from the scaler
    # These are CRITICAL for production systems (HMG4, implementation script) to work correctly
    scaler_params = {
        'max_value': float(scaler.data_max_[0]),
        'min_value': float(scaler.data_min_[0]),
        'feature_range': [-1, 1],  # The target range used during training
        'normalization_method': 'MinMaxScaler',
        'description': 'Normalization parameters calculated from training data only. These values MUST be used by all downstream systems to ensure consistent data processing.',
        'usage_instructions': {
            'input_normalization': f'(value - {scaler.data_min_[0]}) / ({scaler.data_max_[0]} - {scaler.data_min_[0]}) * 2 - 1',
            'output_denormalization': f'(predicted_value + 1) / 2 * ({scaler.data_max_[0]} - {scaler.data_min_[0]}) + {scaler.data_min_[0]}',
            'python_example': f'# Normalize input: normalized = (value - {scaler.data_min_[0]}) / ({scaler.data_max_[0]} - {scaler.data_min_[0]}) * 2 - 1\n# Denormalize output: denormalized = (predicted + 1) / 2 * ({scaler.data_max_[0]} - {scaler.data_min_[0]}) + {scaler.data_min_[0]}'
        },
        'training_data_stats': {
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'test_samples': len(test_data),
            'original_data_range': [float(train_data.min()), float(train_data.max())],
            'scaled_data_range': [-1.0, 1.0]
        }
    }
    
    # Save normalization parameters to a JSON file
    params_path = os.path.join(args.output_dir, 'scaler_params.json')
    with open(params_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    
    # Also save a simplified version for easy integration
    simple_params_path = os.path.join(args.output_dir, 'normalization_params_simple.json')
    simple_params = {
        'max_value': float(scaler.data_max_[0]),
        'min_value': float(scaler.data_min_[0])
    }
    with open(simple_params_path, 'w') as f:
        json.dump(simple_params, f, indent=4)
    
    # NEW: Save differencing parameters for production systems
    differencing_params_path = os.path.join(args.output_dir, 'differencing_params.json')
    differencing_params = {
        'daily_lag': 96,
        'description': 'Daily seasonal differencing parameters for 15-minute data (24h * 4 steps per hour)',
        'differencing_formula': 'differenced_value = value[t] - value[t-96]',
        'inverse_formula': 'reconstructed_value = predicted_change + value[t-96]',
        'python_example': '# Apply differencing: differenced = value[t] - value[t-96]\n# Reconstruct: reconstructed = predicted_change + value[t-96]',
        'notes': 'The model now predicts changes rather than absolute values. This removes strong daily seasonality patterns.',
        'production_requirements': [
            'Apply daily differencing (lag=96) to input data before normalization',
            'Model predicts the change from 96 steps ago',
            'Add predicted change to value from 96 steps ago to get final forecast',
            'This inverse differencing must be done in both evaluation and production'
        ]
    }
    with open(differencing_params_path, 'w') as f:
        json.dump(differencing_params, f, indent=4)
    
    logging.info(f"Normalization parameters saved to {params_path}")
    logging.info(f"Simplified parameters saved to {simple_params_path}")
    logging.info(f"NEW: Differencing parameters saved to {differencing_params_path}")
    logging.info(f"  - max_value: {scaler_params['max_value']}")
    logging.info(f"  - min_value: {scaler_params['min_value']}")
    logging.info(f"  - feature_range: {scaler_params['feature_range']}")
    logging.info(f"  - daily_lag: {differencing_params['daily_lag']}")
    logging.info("")
    logging.info("CRITICAL: These normalization AND differencing parameters MUST be used by:")
    logging.info("  1. Your implementation script (update config.ini)")
    logging.info("  2. HMG4 production system (update critical_load_models.csv)")
    logging.info("  3. Any other system that uses this trained model")
    logging.info("")
    logging.info("NEW: The model now predicts CHANGES, not absolute values!")
    logging.info("  - Input data must be differenced (lag=96) before normalization")
    logging.info("  - Model output must be inverse-differenced to get final forecasts")
    logging.info("  - This is critical for correct operation in production!")
    logging.info("")
    logging.info("Without these values, the model will fail because input scaling will be incorrect!")
    logging.info("="*50)

if __name__ == "__main__":
    # Setup command-line argument parsing for SOTA features
    parser = argparse.ArgumentParser(description="SOTA CNN-LSTM Time Series Trainer with Advanced Features")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the training data CSVs.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save models and metrics.')
    parser.add_argument('--logs_dir', type=str, required=True, help='Directory to save logs.')
    parser.add_argument('--dataset_kind', type=str, default='electric', choices=['electric','solar'], help='Dataset type: solar keeps nighttime zeros; electric treats zeros as missing.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (matches SLURM scripts, with early stopping).')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size PER GPU (optimized for 80GB A100s, matches SLURM scripts). Reduce to 128 if OOM occurs.')
    parser.add_argument('--use_tuner', action='store_true', help='Use Keras Tuner for hyperparameter optimization.')
    parser.add_argument('--max_trials', type=int, default=40, help='Maximum number of hyperparameter trials (increased from 15 to leverage expanded search space).')
    parser.add_argument('--epochs_per_trial', type=int, default=30, help='Number of epochs per hyperparameter trial (increased from 25 for better convergence).')
    parser.add_argument('--enable_augmentation', action='store_true', default=True, help='Enable advanced data augmentation.')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Enable mixed precision training.')
    parser.add_argument('--xla_compilation', action='store_true', default=True, help='Enable XLA compilation.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (optimized for 200 epochs training).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--model_type', type=str, default='cnn_lstm_attention', 
                       choices=['cnn_lstm_attention', 'lstm_only'], help='Model architecture type.')
    parser.add_argument("--export_cpu", action="store_true",
                    help="Also export a CPU-compatible SavedModel (no CuDNN ops) after training")
    parser.add_argument("--seq_len", type=int, default=96, help="Sequence length / window size (96 = 24h of 15-min steps)")
    parser.add_argument("--cpu_only", action="store_true",
                    help="Force CPU-only execution (ignores GPUs even if visible).")
    
    args = parser.parse_args()
    main(args)