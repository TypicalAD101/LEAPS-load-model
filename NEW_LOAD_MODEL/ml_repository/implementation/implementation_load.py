# implementation_load.py — NEW LOAD MODEL IMPLEMENTATION / INFERENCE SCRIPT
# Forked from implementation.py
# KEY CHANGES:
#   1. REMOVED daily seasonal differencing entirely
#   2. ADDED compute_time_features() for 7 cyclical time features
#   3. Model input is (96, 8) — [scaled_load, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend]
#   4. Model output is (96, 1) — predicted scaled load for next 96 steps
#   5. SINGLE forward pass (seq2seq) — NO recursive forecasting, NO error accumulation
#   6. MinMaxScaler on RAW load values (not differenced)
#   7. Kept wide report format (Input / Actual / Predicted / Error)

import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(tf.config.list_physical_devices("GPU"))
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import configparser
import logging


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def dbg_stats(tag, s):
    """Print debug statistics for a numeric series."""
    s = pd.Series(np.asarray(s).reshape(-1))
    s = pd.to_numeric(s, errors="coerce")

    n = len(s)
    n_nan = int(s.isna().sum())
    n_zero = int((s == 0).sum())
    non_nan = s.dropna()

    nan_rate = (n_nan / n) if n else 0.0
    zero_rate = (n_zero / n) if n else 0.0

    msg = (
        f"[DBG] {tag}: n={n} nan={n_nan} ({nan_rate:.3%}) "
        f"zeros={n_zero} ({zero_rate:.3%}) "
    )

    if len(non_nan) > 0:
        msg += (
            f"min={float(non_nan.min()):.4f} "
            f"p01={float(np.nanpercentile(non_nan,1)):.4f} "
            f"p50={float(np.nanpercentile(non_nan,50)):.4f} "
            f"p99={float(np.nanpercentile(non_nan,99)):.4f} "
            f"max={float(non_nan.max()):.4f} "
            f"mean={float(non_nan.mean()):.4f} std={float(non_nan.std()):.4f}"
        )
    else:
        msg += "ALL_NAN"

    print(msg)


def compute_time_features(timestamps):
    """
    Compute 7 cyclical time features from an array of datetime timestamps.
    MUST match the function in train_load.py exactly.

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
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger("implementation_load")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================================
# SAVEDMODEL INFERENCE WRAPPER
# =============================================================================
class SavedModelInferenceWrapper:
    """Wraps a TensorFlow SavedModel to provide a Keras-like .predict() interface."""

    def __init__(self, infer_func, inp_spec, out_spec, input_name):
        self.infer_func = infer_func
        self.input_spec = inp_spec
        self.output_spec = out_spec
        self.input_name = input_name
        self.input_shape = (None,) + tuple(inp_spec.shape[1:])
        self.output_shape = (None,) + tuple(out_spec.shape[1:])

    def predict(self, x, verbose=0):
        """Make predictions using the SavedModel signature."""
        x_tensor = tf.convert_to_tensor(x, dtype=self.input_spec.dtype)
        result = self.infer_func(**{self.input_name: x_tensor})
        if isinstance(result, dict):
            output = list(result.values())[0]
        else:
            output = result
        return output.numpy()


# =============================================================================
# MAIN IMPLEMENTATION CLASS
# =============================================================================
class implementation_load:
    """
    Load model implementation / inference class.

    Receives a trained SavedModel (or .keras/.h5) that was trained with
    train_load.py (no differencing, 8-channel input with time features).

    Performs:
        - CSV loading and format conversion
        - MinMaxScaler normalization using config.ini data_min / data_max (RAW values)
        - Time feature construction from timestamps
        - Single-pass seq2seq prediction (96 → 96)
        - Sliding window simulation with report generation
    """

    def __init__(self, model_path, test_data_path, config_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.config_path = config_path
        self.test_data = None
        self.model = None
        self.data_max = None
        self.data_min = None
        self.feature_range_min = -1.0
        self.feature_range_max = 1.0
        self.sequence_length = 96
        self.n_input_features = 8
        self.dataset_kind = "load"
        # Stores the raw load values and timestamps aligned 1:1
        self.raw_values = None      # shape (N,)
        self.timestamps = None      # pd.DatetimeIndex of length N
        self.time_features = None   # shape (N, 7)

    # -----------------------------------------------------------------
    # CSV CONVERSION (same as original — supports multiple formats)
    # -----------------------------------------------------------------
    def convert_csv_format(self, csv_path):
        """
        Convert CSV formats to expected Load_Profile-like format.
        Supported:
            1) Solar NREL format: LocalTime, Power(MW)
            2) Simple time-series format: time/value or Time/Value
            3) Already Load_Profile format (Day Month Year Time ... controllableLoadForecast_0)
        """
        df = pd.read_csv(csv_path)
        cols_lower = [c.strip().lower() for c in df.columns]
        col_map = {c.strip().lower(): c for c in df.columns}

        print(f"\n[DBG] Loaded CSV: {csv_path}")
        print(f"[DBG] Columns after strip: {df.columns.tolist()}")
        print(f"[DBG] Head:\n{df.head(3)}")
        print(f"[DBG] Tail:\n{df.tail(3)}")

        for c in df.columns:
            if c.strip().lower() in ["value", "generation", "mw", "power(mw)", "power_mw", "power"]:
                tmp = pd.to_numeric(df[c], errors="coerce")
                print(f"[DBG] col='{c}' numeric_nonnull={tmp.notna().sum()}/{len(tmp)} "
                      f"zeros={(tmp==0).sum()} zero_rate={(tmp==0).mean():.6f}")

        # ---- Case A: Solar NREL style ----
        if ('localtime' in cols_lower) and (
            ('power(mw)' in cols_lower) or ('power_mw' in cols_lower) or ('power' in cols_lower)
        ):
            time_col = col_map['localtime']
            if 'power(mw)' in cols_lower:
                val_col = col_map['power(mw)']
            elif 'power_mw' in cols_lower:
                val_col = col_map['power_mw']
            else:
                val_col = col_map['power']

            dt = pd.to_datetime(df[time_col], errors='coerce')
            vals = pd.to_numeric(df[val_col], errors='coerce')

            result_df = pd.DataFrame({
                'Day': dt.dt.day.astype('Int64'),
                'Month': dt.dt.month.astype('Int64'),
                'Year': dt.dt.year.astype('Int64'),
                'Time': dt.dt.strftime('%H:%M:%S'),
                'controllableLoadForecast_0': vals,
                'criticalLoadForecast_0': 0.0
            })
            result_df = result_df.dropna(subset=['Day', 'Month', 'Year', 'Time'])
            return result_df

        # ---- Case B: Simple time/value style ----
        if (('time' in cols_lower) or ('timestamp' in cols_lower) or ('datetime' in cols_lower)) and \
           (('value' in cols_lower) or ('generation' in cols_lower) or ('mw' in cols_lower)):
            for k in ('time', 'timestamp', 'datetime'):
                if k in cols_lower:
                    time_col = col_map[k]
                    break
            for k in ('value', 'generation', 'mw'):
                if k in cols_lower:
                    val_col = col_map[k]
                    break

            dt = pd.to_datetime(df[time_col], errors='coerce')
            vals = pd.to_numeric(df[val_col], errors='coerce')

            result_df = pd.DataFrame({
                'Day': dt.dt.day.astype('Int64'),
                'Month': dt.dt.month.astype('Int64'),
                'Year': dt.dt.year.astype('Int64'),
                'Time': dt.dt.strftime('%H:%M:%S'),
                'controllableLoadForecast_0': vals,
                'criticalLoadForecast_0': 0.0
            })
            result_df = result_df.dropna(subset=['Day', 'Month', 'Year', 'Time'])
            return result_df

        # ---- Case C: Already Load_Profile style ----
        return df

    # -----------------------------------------------------------------
    # LOAD DEPENDENCIES
    # -----------------------------------------------------------------
    def load_dependencies(self):
        """Load model, config, and prepare test data (no differencing)."""
        os.makedirs("reports", exist_ok=True)
        logger.info(f"Loading dependencies: model={self.model_path}, "
                    f"data={self.test_data_path}, config={self.config_path}")

        # 1. Load and convert CSV
        self.test_data = self.convert_csv_format(self.test_data_path)

        # 2. Read config parameters
        config = configparser.ConfigParser()
        config.read(self.config_path)

        self.data_max = float(config['Model']['data_max'])
        self.data_min = float(config['Model']['data_min'])
        self.dataset_kind = config.get('Data', 'dataset_kind', fallback='load').strip().lower()
        self.n_input_features = int(config['Model'].get('n_input_features', '8'))

        try:
            self.feature_range_min = float(config['Model'].get('feature_range_min', '-1.0'))
            self.feature_range_max = float(config['Model'].get('feature_range_max', '1.0'))
            self.sequence_length = int(config['Model'].get('sequence_length', '96'))
        except (TypeError, ValueError) as e:
            logger.warning(f"Error reading config parameters, using defaults: {e}")

        print(f"[DBG] dataset_kind={self.dataset_kind} | data_min={self.data_min} | data_max={self.data_max}")
        print(f"[DBG] n_input_features={self.n_input_features} | sequence_length={self.sequence_length}")
        print(f"[DBG] NOTE: data_min / data_max are min / max of RAW load values (NO differencing).")

        # 3. Extract raw load values
        x_raw = pd.to_numeric(self.test_data.iloc[:, 4], errors="coerce")
        dbg_stats("controllable_raw", x_raw)

        # Store pristine copy
        self.raw_values = x_raw.values.copy().astype(np.float64)

        # 4. Reconstruct timestamps from Day/Month/Year/Time columns
        try:
            dt = pd.to_datetime(
                self.test_data["Year"].astype(str) + "-" +
                self.test_data["Month"].astype(str) + "-" +
                self.test_data["Day"].astype(str) + " " +
                self.test_data["Time"].astype(str),
                errors="coerce"
            )
            self.timestamps = pd.DatetimeIndex(dt)
            logger.info(f"Reconstructed {self.timestamps.notna().sum()} timestamps "
                        f"({self.timestamps.isna().sum()} NaT)")
            # Quick timestep check
            deltas = dt.sort_values().diff().dropna()
            if len(deltas) > 0:
                mode_delta = deltas.mode().iloc[0]
                print(f"[DBG] inferred_timestep_mode={mode_delta}")
        except Exception as e:
            logger.warning(f"Timestamp reconstruction failed: {e}")
            # Fallback: create synthetic 15-min timestamps
            logger.info("Falling back to synthetic 15-min timestamps starting 2024-01-01")
            self.timestamps = pd.date_range(
                start="2024-01-01", periods=len(self.raw_values), freq="15min"
            )

        # 5. Compute time features for full dataset
        self.time_features = compute_time_features(self.timestamps)
        logger.info(f"Time features shape: {self.time_features.shape}")

        # 6. Zero handling for load data
        print(f"[DBG] raw zeros={(x_raw == 0).sum()} zero_rate={(x_raw == 0).mean():.6f} "
              f"NaNs={x_raw.isna().sum()}")
        if self.dataset_kind == "load":
            # Replace 0 with NaN (matches training behavior)
            mask_zero = self.raw_values == 0
            if mask_zero.sum() > 0:
                logger.info(f"Replacing {mask_zero.sum()} zeros with NaN (load dataset)")
                self.raw_values[mask_zero] = np.nan

        dbg_stats("raw_values_after_zero_handling", self.raw_values)

        logger.info(f"Normalization params → data_max: {self.data_max}, data_min: {self.data_min}, "
                    f"feature_range: [{self.feature_range_min}, {self.feature_range_max}]")
        logger.info(f"NO differencing applied — model trained on raw scaled values.")

        # Validate critical normalization parameters
        if self.data_max is None or self.data_min is None:
            raise ValueError("Missing critical normalization parameters (data_min / data_max)")

        # 7. Load the model
        self._load_model()

        # 8. Verify model architecture
        self.verify_model_architecture()

        logger.info("Dependencies loaded successfully.")

    # -----------------------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------------------
    def _load_model(self):
        """Load model — supports SavedModel directory, .keras, .h5"""
        if os.path.isdir(self.model_path):
            self._load_savedmodel()
        elif self.model_path.endswith('.keras') or self.model_path.endswith('.h5'):
            self._load_keras_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")

    def _load_savedmodel(self):
        """Load TensorFlow SavedModel (CPU-compatible)."""
        logger.info(f"Loading SavedModel from directory: {self.model_path}")
        try:
            loaded = tf.saved_model.load(self.model_path)
            if 'serving_default' not in loaded.signatures:
                raise ValueError("SavedModel does not have a 'serving_default' signature")

            infer = loaded.signatures['serving_default']
            input_spec = list(infer.structured_input_signature[1].values())[0]
            output_spec = list(infer.structured_outputs.values())[0]
            input_key = list(infer.structured_input_signature[1].keys())[0]

            logger.info(f"SavedModel input shape: {input_spec.shape}, dtype: {input_spec.dtype}")
            logger.info(f"SavedModel output shape: {output_spec.shape}, dtype: {output_spec.dtype}")

            self.model = SavedModelInferenceWrapper(infer, input_spec, output_spec, input_key)
            logger.info(f"SavedModel loaded (CPU wrapper). "
                        f"Input: {self.model.input_shape}, Output: {self.model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"SavedModel loading failed: {e}")

    def _load_keras_model(self):
        """Load .keras or .h5 model file."""
        import keras
        logger.info(f"Loading Keras model (version {keras.__version__}): {self.model_path}")

        custom_objects = {}
        try:
            from keras.initializers import Orthogonal, GlorotUniform, Zeros, Ones
            from keras.layers import (MultiHeadAttention, Conv1D, BatchNormalization,
                                      LSTM, Dense, Dropout, MaxPooling1D)

            class CompatibleOrthogonal(Orthogonal):
                def __init__(self, gain=1.0, seed=None, **kwargs):
                    super().__init__(gain=gain, **kwargs)

            class CompatibleGlorotUniform(GlorotUniform):
                def __init__(self, seed=None, **kwargs):
                    super().__init__(**kwargs)

            custom_objects.update({
                'Orthogonal': CompatibleOrthogonal,
                'GlorotUniform': CompatibleGlorotUniform,
                'Zeros': Zeros,
                'Ones': Ones,
            })
        except Exception as e:
            logger.warning(f"Could not register custom objects: {e}")

        try:
            if self.model_path.endswith('.keras'):
                try:
                    self.model = keras.models.load_model(
                        self.model_path, compile=False, custom_objects=custom_objects)
                except Exception:
                    self.model = keras.models.load_model(
                        self.model_path, compile=False, safe_mode=False)
            else:
                try:
                    self.model = keras.models.load_model(
                        self.model_path, compile=False, safe_mode=False,
                        custom_objects=custom_objects)
                except Exception:
                    self.model = keras.models.load_model(
                        self.model_path, compile=False, custom_objects=custom_objects)
            logger.info(f"Keras model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Keras model loading failed: {e}")

    # -----------------------------------------------------------------
    # MODEL VERIFICATION
    # -----------------------------------------------------------------
    def verify_model_architecture(self):
        """Verify model input/output shapes match expectations."""
        logger.info("=" * 60)
        logger.info("VERIFYING MODEL ARCHITECTURE (LOAD MODEL — 8-channel input)")
        logger.info("=" * 60)

        expected_input = (None, self.sequence_length, self.n_input_features)
        expected_output = (None, self.sequence_length, 1)

        actual_input = self.model.input_shape
        actual_output = self.model.output_shape

        logger.info(f"Expected input shape:  {expected_input}")
        logger.info(f"Actual   input shape:  {actual_input}")
        logger.info(f"Expected output shape: {expected_output}")
        logger.info(f"Actual   output shape: {actual_output}")

        # Check input features dimension
        if actual_input[-1] != self.n_input_features:
            logger.error(f"INPUT SHAPE MISMATCH! Model expects {actual_input[-1]} features, "
                         f"but config says n_input_features={self.n_input_features}")
        else:
            logger.info("Input feature dimension matches ✓")

        # Check if model has GlobalAveragePooling (architecture anti-pattern)
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if 'GlobalAveragePooling' in str(type(layer)):
                    logger.error(f"CRITICAL: Found GlobalAveragePooling layer: {layer.name}")

        return True

    # -----------------------------------------------------------------
    # NORMALIZATION HELPERS
    # -----------------------------------------------------------------
    def _normalize(self, values):
        """MinMaxScaler normalize: raw → [-1, 1] using training data_min / data_max."""
        a = self.feature_range_min
        b = self.feature_range_max
        return a + (values - self.data_min) * (b - a) / (self.data_max - self.data_min)

    def _denormalize(self, scaled):
        """Inverse MinMaxScaler: [-1, 1] → raw scale."""
        a = self.feature_range_min
        b = self.feature_range_max
        return self.data_min + (scaled - a) * (self.data_max - self.data_min) / (b - a)

    # -----------------------------------------------------------------
    # SEQ2SEQ PREDICTION (REPLACES RECURSIVE FORECASTING)
    # -----------------------------------------------------------------
    def _perform_seq2seq_forecast(self, input_load_scaled, output_time_features):
        """
        Perform a single forward pass through the seq2seq model.

        Args:
            input_load_scaled: np.ndarray of shape (96,) — previous day's scaled load values
            output_time_features: np.ndarray of shape (96, 7) — time features for the OUTPUT window

        Returns:
            predictions_raw: np.ndarray of shape (96,) — predicted load in original scale
            predictions_scaled: np.ndarray of shape (96,) — predicted load in scaled space
        """
        # Construct the 8-channel input: [scaled_load, 7 time features]
        # X[:, 0]   = scaled load history (previous day)
        # X[:, 1:8] = time features for the prediction window (next day)
        model_input = np.zeros((1, self.sequence_length, self.n_input_features), dtype=np.float32)
        model_input[0, :, 0] = input_load_scaled
        model_input[0, :, 1:] = output_time_features

        # Forward pass
        pred = self.model.predict(model_input, verbose=0)
        pred_arr = np.asarray(pred).reshape(self.sequence_length)  # shape (96,)

        # Denormalize predictions back to raw scale
        predictions_raw = self._denormalize(pred_arr)

        # Clamp negative values to zero (load can't be negative)
        predictions_raw = np.maximum(predictions_raw, 0.0)

        return predictions_raw, pred_arr

    # -----------------------------------------------------------------
    # SIMULATION — SLIDING WINDOW
    # -----------------------------------------------------------------
    def run_simulation(self):
        """
        Run sliding window simulation over the test data.

        For each window position t:
            - Input: raw_values[t-96 : t] (previous day)  → normalize → model input load channel
            - Time features: computed from timestamps[t : t+96] (next day)
            - Output: model predicts 96 steps in one pass → denormalized to raw scale
            - Actuals: raw_values[t : t+96]
        """
        logger.info("Starting sliding window simulation (seq2seq, no recursion)...")
        all_results = []

        n = len(self.raw_values)

        # We need at least 96 history + 96 forecast horizon
        # Window slides by 1 timestep; first prediction starts at index 96
        max_start = n - self.sequence_length  # last index where we can still get 96 actuals
        total_windows = max_start - self.sequence_length + 1

        if total_windows <= 0:
            logger.warning("Not enough data for even a single rolling window.")
            return None

        log_interval = max(1, total_windows // 10)
        logger.info(f"Total rolling windows: {total_windows} (sliding by 1 timestep)")

        for idx in range(total_windows):
            t = self.sequence_length + idx  # prediction window starts at index t

            if idx == 0:
                logger.info(f"Processing first window. input=[{t-self.sequence_length}:{t}], "
                            f"predict=[{t}:{t+self.sequence_length}]")
            if (idx + 1) % log_interval == 0 or (idx + 1) == total_windows:
                logger.info(f"Progress: {idx+1}/{total_windows} "
                            f"({(idx+1)/total_windows*100:.1f}%)")

            # --- Input: previous day's load ---
            input_raw = self.raw_values[t - self.sequence_length : t]

            # Handle NaN in input (forward-fill then back-fill)
            input_series = pd.Series(input_raw)
            if input_series.isna().any():
                input_series = input_series.ffill().bfill()
                if input_series.isna().any():
                    # Last resort: fill remaining NaN with data_min (safe fallback)
                    input_series = input_series.fillna(self.data_min)
            input_clean = input_series.values

            # Normalize input load
            input_scaled = self._normalize(input_clean).astype(np.float32)

            if idx < 3:
                dbg_stats("input_raw", input_raw)
                dbg_stats("input_scaled", input_scaled)

            # --- Time features for the OUTPUT window ---
            output_tf = self.time_features[t : t + self.sequence_length]
            if len(output_tf) < self.sequence_length:
                # Pad with last row if needed (edge case at end of data)
                pad = np.tile(output_tf[-1:], (self.sequence_length - len(output_tf), 1))
                output_tf = np.vstack([output_tf, pad])

            # --- Predict ---
            predictions_raw, predictions_scaled = self._perform_seq2seq_forecast(
                input_scaled, output_tf
            )

            # --- Actuals ---
            actuals_end = min(t + self.sequence_length, n)
            actuals = self.raw_values[t : actuals_end].copy()
            if len(actuals) < self.sequence_length:
                actuals = np.concatenate([actuals,
                    np.full(self.sequence_length - len(actuals), np.nan)])

            if idx < 3:
                dbg_stats("predictions_raw", predictions_raw)
                dbg_stats("actuals", actuals)

            all_results.append({
                'timestamp': t,
                'actuals': actuals,
                'predictions': predictions_raw,
                'raw_input_window': input_clean.copy(),
                'window_index': idx,
            })

            if idx == total_windows - 1:
                logger.info(f"Finished last window at index {t}")

        logger.info("Simulation complete. Generating report...")
        return self._generate_report(all_results)

    # -----------------------------------------------------------------
    # REPORT GENERATION
    # -----------------------------------------------------------------
    def _generate_report(self, results):
        """Generate performance reports (long + wide format)."""
        logger.info("Generating performance reports...")

        all_actuals = []
        all_preds = []
        horizon_errors = {4: [], 12: [], 24: [], 48: [], 96: []}  # 1h, 3h, 6h, 12h, 24h

        for entry in results:
            ts = entry['timestamp']
            preds = entry['predictions']
            actuals = entry['actuals']

            for horizon in [4, 12, 24, 48, 96]:
                if len(actuals) >= horizon and len(preds) >= horizon:
                    a_h = np.array(actuals[:horizon])
                    p_h = np.array(preds[:horizon])
                    valid = ~np.isnan(a_h)
                    if valid.sum() > 0:
                        horizon_errors[horizon].append(
                            mean_absolute_error(a_h[valid], p_h[valid]))

            for i in range(min(len(actuals), len(preds))):
                if not np.isnan(actuals[i]):
                    all_actuals.append(actuals[i])
                    all_preds.append(preds[i])

        # Overall metrics
        if all_actuals:
            mae = mean_absolute_error(all_actuals, all_preds)
            rmse = float(np.sqrt(mean_squared_error(all_actuals, all_preds)))
        else:
            mae = rmse = float('nan')

        logger.info(f'Overall MAE: {mae:.4f}')
        logger.info(f'Overall RMSE: {rmse:.4f}')

        logger.info('Average MAE by forecast horizon:')
        for horizon, errors in horizon_errors.items():
            if errors:
                logger.info(f'  First {horizon} steps ({horizon//4}h): MAE = {np.mean(errors):.4f}')
            else:
                logger.info(f'  First {horizon} steps ({horizon//4}h): MAE = N/A')

        # Long-format CSV
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        simplified_rows = []
        for i in range(len(self.raw_values)):
            prediction_val = None
            for entry in results:
                if entry['timestamp'] <= i < entry['timestamp'] + len(entry['predictions']):
                    pred_idx = i - entry['timestamp']
                    prediction_val = entry['predictions'][pred_idx]
                    break
            if prediction_val is None:
                prediction_val = self.raw_values[i]  # Fallback
            simplified_rows.append({
                'timestamp': i,
                'actual': self.raw_values[i],
                'prediction': prediction_val,
                'error': prediction_val - self.raw_values[i]
            })

        df_long = pd.DataFrame(simplified_rows)
        long_csv = f'reports/performance_report_{model_name}.csv'
        df_long.to_csv(long_csv, index=False)
        logger.info(f'Long-format report saved to {long_csv} ({len(simplified_rows)} rows)')

        # Wide-format report
        self._generate_wide_format_report(results, model_name)

        return {
            'model_name': model_name,
            'mae_1hr': np.mean(horizon_errors[4]) if horizon_errors[4] else np.nan,
            'mae_3hr': np.mean(horizon_errors[12]) if horizon_errors[12] else np.nan,
            'mae_6hr': np.mean(horizon_errors[24]) if horizon_errors[24] else np.nan,
            'mae_12hr': np.mean(horizon_errors[48]) if horizon_errors[48] else np.nan,
            'mae_overall': mae,
        }

    def _generate_wide_format_report(self, results, model_name):
        """Generate wide-format report: Input / Actual / Predicted / Error per rolling window."""
        logger.info("Generating wide-format report...")

        all_windows_dfs = []

        for window_idx, entry in enumerate(results):
            row_labels = [f"t+{i}" for i in range(self.sequence_length)]
            metric_labels = ["MAE_Overall", "MAE_3h", "MAE_6h", "MAE_12h", "MAE_24h"]
            full_index = row_labels + metric_labels

            column_headers = ["Input", "Actual", "Predicted", "Error"]
            df_window = pd.DataFrame(index=full_index, columns=column_headers, dtype=object)

            actuals = entry['actuals']
            preds = entry['predictions']
            input_window = entry['raw_input_window']

            for t in range(self.sequence_length):
                df_window.loc[f"t+{t}", "Input"] = (
                    input_window[t] if t < len(input_window) else np.nan)

                if t < len(actuals) and not np.isnan(actuals[t]):
                    df_window.loc[f"t+{t}", "Actual"] = actuals[t]
                    df_window.loc[f"t+{t}", "Error"] = preds[t] - actuals[t]
                else:
                    df_window.loc[f"t+{t}", "Actual"] = np.nan
                    df_window.loc[f"t+{t}", "Error"] = np.nan

                df_window.loc[f"t+{t}", "Predicted"] = preds[t]

            def safe_mae(a, p):
                a_arr, p_arr = np.array(a, dtype=float), np.array(p, dtype=float)
                valid = ~np.isnan(a_arr)
                return mean_absolute_error(a_arr[valid], p_arr[valid]) if valid.sum() > 0 else np.nan

            df_window.loc["MAE_Overall", "Predicted"] = safe_mae(actuals, preds)
            if len(preds) >= 12:
                df_window.loc["MAE_3h", "Predicted"] = safe_mae(actuals[:12], preds[:12])
            if len(preds) >= 24:
                df_window.loc["MAE_6h", "Predicted"] = safe_mae(actuals[:24], preds[:24])
            if len(preds) >= 48:
                df_window.loc["MAE_12h", "Predicted"] = safe_mae(actuals[:48], preds[:48])
            if len(preds) >= self.sequence_length:
                df_window.loc["MAE_24h", "Predicted"] = safe_mae(
                    actuals[:self.sequence_length], preds[:self.sequence_length])

            df_window.columns = pd.MultiIndex.from_product(
                [[f"Rolling Window {window_idx}"], df_window.columns])
            all_windows_dfs.append(df_window)

        if not all_windows_dfs:
            logger.warning("No data available to generate the wide-format report.")
            return None

        final_df = pd.concat(all_windows_dfs, axis=1)
        output_file = f"reports/performance_report_{model_name}_wide.csv"
        final_df.to_csv(output_file)
        logger.info(f"Wide-format report saved to {output_file}")
        return output_file


# =============================================================================
# ENTRY POINTS
# =============================================================================
def run_all_models_implementation():
    """Run implementation for all models in the models/ directory."""
    logger.info("Starting implementation for all models...")

    config = configparser.ConfigParser()
    config.read('config.ini')

    models_dir = "models"
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return

    impl_dir = config['Paths']['impl_path']
    if not os.path.exists(impl_dir):
        logger.error(f"Implementation directory not found: {impl_dir}")
        return

    # Discover models (SavedModel dirs, .h5, .keras)
    model_entries = []
    for entry in os.listdir(models_dir):
        full = os.path.join(models_dir, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "saved_model.pb")):
            model_entries.append((entry, full))
        elif os.path.isfile(full) and (entry.endswith('.h5') or entry.endswith('.keras')):
            model_entries.append((entry, full))

    csv_files = [(f, os.path.join(impl_dir, f))
                 for f in os.listdir(impl_dir) if f.endswith('.csv')]

    logger.info(f"Found {len(model_entries)} model(s), {len(csv_files)} CSV file(s)")

    if not model_entries or not csv_files:
        logger.warning("Need at least one model and one CSV.")
        return

    all_results = []
    for model_name, model_path in model_entries:
        for csv_name, csv_path in csv_files:
            try:
                logger.info(f"Running: model={model_name}, data={csv_name}")
                impl = implementation_load(model_path, csv_path, 'config.ini')
                impl.load_dependencies()
                metrics = impl.run_simulation()
                if metrics:
                    metrics['model_name'] = (
                        f"{os.path.splitext(model_name)[0]}_{os.path.splitext(csv_name)[0]}")
                    all_results.append(metrics)
            except Exception as e:
                logger.error(f"Error: {model_name} + {csv_name}: {e}")
                continue

    if all_results:
        generate_comparative_summary(all_results)
    else:
        logger.warning("No successful model runs to compare.")


def generate_comparative_summary(all_results):
    """Generate comparative summary CSV."""
    summary = []
    for r in all_results:
        summary.append({
            'model': r['model_name'],
            'MAE 1HR': r['mae_1hr'],
            'MAE 3HR': r['mae_3hr'],
            'MAE 6HR': r['mae_6hr'],
            'MAE 12HR': r['mae_12hr'],
            'MAE Overall': r['mae_overall'],
        })
    df = pd.DataFrame(summary)
    output = "reports/comparative_summary.csv"
    df.to_csv(output, index=False)
    logger.info(f"Comparative summary saved to {output}")
    logger.info(f"Summary:\n{df.to_string(index=False)}")
    return output
