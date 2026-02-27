import os
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(tf.config.list_physical_devices("GPU"))
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import configparser
import logging
import sqlite3


def dbg_stats(tag, s):
    import numpy as np
    import pandas as pd

    # Accept numpy arrays, lists, pandas Series, etc.
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


logger = logging.getLogger("implementation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class implementation:
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
        self.daily_lag = 96
        self.sequence_length = 96
        self.original_values = None
        self.differenced_to_original_mapping = {}
        self.dataset_kind = "load"
        # ----- START FIX: Predictions are being offset by 96, adding the following to handle that issue -----
        self.diff_orig_idx = None   # numpy array mapping differenced-row-position -> original index
        # ----- END FIX -----

    def convert_csv_format(self, csv_path):
        """
        Convert CSV formats to expected Load_Profile-like format.

        Supported:
        1) Solar NREL format: LocalTime, Power(MW)
        2) Simple time-series format: time/value or Time/Value
        3) Already Load_Profile format (Day Month Year Time ... controllableLoadForecast_0 criticalLoadForecast_0)
        """
        df = pd.read_csv(csv_path)
        cols_lower = [c.strip().lower() for c in df.columns]
        col_map = {c.strip().lower(): c for c in df.columns}

        print(f"\n[DBG] Loaded CSV: {csv_path}")
        print(f"[DBG] Columns after strip: {df.columns.tolist()}")
        print(f"[DBG] Head:\n{df.head(3)}")
        print(f"[DBG] Tail:\n{df.tail(3)}")
        # quick numeric scan on the likely value-like columns (helps catch wrong col mapping)
        for c in df.columns:
            if c.strip().lower() in ["value", "generation", "mw", "power(mw)", "power_mw", "power"]:
                tmp = pd.to_numeric(df[c], errors="coerce")
                print(f"[DBG] col='{c}' numeric_nonnull={tmp.notna().sum()}/{len(tmp)} zeros={(tmp==0).sum()} zero_rate={(tmp==0).mean():.6f}")

        # ---- Case A: Solar NREL style ----
        # e.g., LocalTime + Power(MW)
        if ('localtime' in cols_lower) and (('power(mw)' in cols_lower) or ('power_mw' in cols_lower) or ('power' in cols_lower)):
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

            # Drop rows with bad timestamps
            result_df = result_df.dropna(subset=['Day', 'Month', 'Year', 'Time'])
            return result_df

        # ---- Case B: Simple time/value style ----
        # e.g., time/value, Time/Value, timestamp/generation, etc.
        if (('time' in cols_lower) or ('timestamp' in cols_lower) or ('datetime' in cols_lower)) and (('value' in cols_lower) or ('generation' in cols_lower) or ('mw' in cols_lower)):
            # time column
            for k in ('time', 'timestamp', 'datetime'):
                if k in cols_lower:
                    time_col = col_map[k]
                    break

            # value column
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

        # ---- Case C: Already Load_Profile style (expected) ----
        # Just return as-is
        return df

    def load_dependencies(self):
        os.makedirs("reports", exist_ok=True)
        logger.info(f"Loading dependencies: model={self.model_path}, data={self.test_data_path}, config={self.config_path}")
        
        # Load and convert CSV if needed
        self.test_data = self.convert_csv_format(self.test_data_path)

        # Read config parameters
        config = configparser.ConfigParser()
        config.read(self.config_path)
        
        # CRITICAL: These parameters MUST match the training scaler parameters
        # Ideally loaded from training outputs: scaler_params.json (min_value, max_value)
        # These represent the min/max of the DIFFERENCED training data, not raw data
        self.data_max = float(config['Model'].get('data_max')) if 'data_max' in config['Model'] else None
        self.data_min = float(config['Model'].get('data_min')) if 'data_min' in config['Model'] else None

        # Optional: allow config.ini to specify dataset kind (solar vs load)
        # Default to "load" if missing (since this implementation is currently being used for load model validation)
        self.dataset_kind = config.get('Data', 'dataset_kind', fallback='load').strip().lower()

        print(f"[DBG] dataset_kind={self.dataset_kind} | data_min={self.data_min} | data_max={self.data_max}")
        # Show whether our min/max are for differenced space (expected) by printing raw stats vs after differencing later
        print("[DBG] NOTE: data_min/data_max are expected to be min/max of DIFFERENCED training data (not raw).")

        print(f"[DBG] Converted test_data shape={self.test_data.shape} columns={self.test_data.columns.tolist()}")

        # controllable load series is column index 4 in your later code (Day,Month,Year,Time,controllable,critical)
        x_raw = pd.to_numeric(self.test_data.iloc[:, 4], errors="coerce")
        dbg_stats("controllable_raw_pre_any_processing", x_raw)

        # Store pristine original series BEFORE any modifications (zero->NaN, differencing, dropna)
        self.original_values = x_raw.values.copy()
        logger.info(f"Stored {len(self.original_values)} pristine original data points.")

        # Time continuity check (for 15-min data)
        if "Time" in self.test_data.columns:
            try:
                dt = pd.to_datetime(
                    self.test_data["Year"].astype(str) + "-" +
                    self.test_data["Month"].astype(str) + "-" +
                    self.test_data["Day"].astype(str) + " " +
                    self.test_data["Time"].astype(str),
                    errors="coerce"
                )
                deltas = dt.sort_values().diff().dropna()
                if len(deltas) > 0:
                    mode_delta = deltas.mode().iloc[0]
                    print(f"[DBG] inferred_timestep_mode={mode_delta}")
            except Exception as e:
                print(f"[DBG] timestep inference failed: {e}")

        # quick zero + NaN checks
        print(f"[DBG] controllable raw: zeros={(x_raw==0).sum()} zero_rate={(x_raw==0).mean():.6f} NaNs={x_raw.isna().sum()}")

        # --- Zero handling (debuggable before/after) ---
        x_before0 = pd.to_numeric(self.test_data.iloc[:, 4], errors="coerce")
        dbg_stats("before_zero_handling", x_before0)

        if self.dataset_kind == "load":
            # Treat 0 as missing ONLY if this matches training behavior
            self.test_data.iloc[:, 4] = self.test_data.iloc[:, 4].replace(0, np.nan)

        x_after0 = pd.to_numeric(self.test_data.iloc[:, 4], errors="coerce")
        dbg_stats("after_zero_handling", x_after0)
        print(f"[DBG] after zero handling: NaNs={x_after0.isna().sum()}")

        # # quick zero + NaN checks
        # print(f"[DBG] controllable raw: zeros={(x_raw==0).sum()} zero_rate={(x_raw==0).mean():.6f} NaNs={x_raw.isna().sum()}")


        # if self.dataset_kind == "load":
        #     # Treat 0 as missing (ONLY for debugging/validation unless you confirm this matches training)
        #     self.test_data.iloc[:, 4] = self.test_data.iloc[:, 4].replace(0, np.nan)
        #     x_after = pd.to_numeric(self.test_data.iloc[:, 4], errors="coerce")
        #     dbg_stats("controllable_after_zero_to_nan", x_after)
        #     print(f"[DBG] after zero->NaN: NaNs={pd.isna(x_after).sum()}")

        
        logger.info(f"Reading test data from: {self.test_data_path}")
        logger.info(f"Loaded test data shape: {self.test_data.shape}")
        logger.info(f"First few rows of test data:\n{self.test_data.head()}\n")

        
        try:
            self.feature_range_min = float(config['Model'].get('feature_range_min', -1.0))
            self.feature_range_max = float(config['Model'].get('feature_range_max', 1.0))
            self.daily_lag = int(config['Model'].get('daily_lag', 96))
            self.sequence_length = int(config['Model'].get('sequence_length', 96))
        except (TypeError, ValueError) as e:
            logger.warning(f"Error reading config parameters, using defaults: {e}")

        logger.info(f"Normalization params -> data_max: {self.data_max}, data_min: {self.data_min}, feature_range: [{self.feature_range_min}, {self.feature_range_max}]")
        logger.info(f"Differencing lag (daily_lag): {self.daily_lag}; sequence_length: {self.sequence_length}")

        # Check if model uses new architecture with extended output range
        self.output_range = float(config['Model'].get('output_range', 1.0))
        if self.output_range > 1.0:
            logger.info(f"Model uses extended output range: [-{self.output_range}, {self.output_range}]")
            logger.info("This indicates linear activation architecture")

        # Validate critical normalization parameters
        if self.data_max is None or self.data_min is None:
            logger.error("CRITICAL: data_max and data_min not set in config!")
            logger.error("These are required for denormalization. Check config.ini")
            raise ValueError("Missing critical normalization parameters")

        # Store original values for inverse differencing
        # self.original_values = self.test_data.iloc[:, 4].values.copy()
        logger.info(f"Stored {len(self.original_values)} original data points for inverse differencing.")
        
        # ----- START PATCH -----
        # since predictions are being offset by 96, patch has been added below
        # Because of this patch, this block becomes invalid because it assumes only the first 96 are dropped
        # # Create mapping from differenced to original indices
        # for i in range(len(self.original_values) - self.daily_lag):
        #     self.differenced_to_original_mapping[i] = i + self.daily_lag
        # ----- END PATCH -----
        
        # Apply differencing
        logger.info(f"Applying daily seasonal differencing (lag={self.daily_lag}) to test data...")
        self.test_data.iloc[:, 4] = self.test_data.iloc[:, 4].diff(self.daily_lag)
        self.test_data.dropna(inplace=True)

        # ----- START PATCH: since predictions are being offset by 96, add the following to handle that issue -----
        # ===== IMPORTANT: mapping from differenced rows -> original series index =====
        # After diff+dropna (and possible 0->NaN), the dataframe index tells us which original
        # timestamp each remaining differenced row corresponds to.
        self.diff_orig_idx = self.test_data.index.to_numpy()

        # Optional sanity print for first few
        print(f"[DBG] diff_orig_idx head: {self.diff_orig_idx[:5]}")
        print(f"[DBG] diff_orig_idx tail: {self.diff_orig_idx[-5:]}")
        # ----- END PATCH -----

        x_diff = pd.to_numeric(self.test_data.iloc[:, 4], errors="coerce")
        dbg_stats("after_diff_and_dropna", x_diff)

        # sanity: how often differenced values exceed training min/max?
        below = (x_diff < self.data_min).mean()
        above = (x_diff > self.data_max).mean()
        print(f"[DBG] differenced outside training min/max: below_min={below:.3%} above_max={above:.3%}")

        logger.info(f"Differencing complete. Dropped initial NaN values. New shape: {self.test_data.shape}")
        
        # Load Keras model
        self._load_keras_model()

        # Verify model architecture and temporal sensitivity
        self.verify_model_architecture()
        self.test_model_temporal_sensitivity()

        logger.info("Dependencies loaded successfully.")

    def _load_keras_model(self):
        """Load Keras 3.x model with clean error handling - supports .h5, .keras, and SavedModel formats"""
        import keras
        logger.info(f"Loading Keras model with version: {keras.__version__}")
        
        # Check if model_path is a directory (SavedModel format)
        if os.path.isdir(self.model_path):
            self._load_savedmodel()
            return
        
        # Check for supported file formats
        if not (self.model_path.endswith('.h5') or self.model_path.endswith('.keras')):
            raise ValueError(f"Only .h5, .keras model files, and SavedModel directories are supported. Got: {self.model_path}")
        
        try:
            # Create custom objects to handle problematic layers and initializers
            custom_objects = {}
            try:
                from keras.initializers import Orthogonal, GlorotUniform, Zeros, Ones
                from keras.layers import MultiHeadAttention, Conv1D, BatchNormalization, LSTM, Dense, Dropout, MaxPooling1D, Add, LayerNormalization, GlobalAveragePooling1D
                
                # Create custom Orthogonal initializer that handles legacy 'seed' parameter
                class CompatibleOrthogonal(Orthogonal):
                    def __init__(self, gain=1.0, seed=None, **kwargs):
                        # Ignore seed parameter if passed
                        super().__init__(gain=gain, **kwargs)
                
                # Create custom GlorotUniform that handles legacy 'seed' parameter  
                class CompatibleGlorotUniform(GlorotUniform):
                    def __init__(self, seed=None, **kwargs):
                        # Ignore seed parameter if passed
                        super().__init__(**kwargs)
                
                # Create custom MultiHeadAttention that handles legacy 'seed' parameter
                class CompatibleMultiHeadAttention(MultiHeadAttention):
                    def __init__(self, num_heads, key_dim, seed=None, **kwargs):
                        # Ignore seed parameter if passed
                        super().__init__(num_heads=num_heads, key_dim=key_dim, **kwargs)
                
                custom_objects.update({
                    # Custom initializers that handle legacy parameters
                    'Orthogonal': CompatibleOrthogonal,
                    'GlorotUniform': CompatibleGlorotUniform, 
                    'Zeros': Zeros,
                    'Ones': Ones,
                    # Layers
                    'MultiHeadAttention': CompatibleMultiHeadAttention,
                    'Conv1D': Conv1D,
                    'BatchNormalization': BatchNormalization,
                    'LSTM': LSTM,
                    'Dense': Dense,
                    'Dropout': Dropout,
                    'MaxPooling1D': MaxPooling1D,
                    'Add': Add,
                    'LayerNormalization': LayerNormalization,
                    'GlobalAveragePooling1D': GlobalAveragePooling1D,
                })
                
                # Try to add DTypePolicy if available
                try:
                    from keras.mixed_precision import DTypePolicy
                    custom_objects['DTypePolicy'] = DTypePolicy
                except ImportError:
                    try:
                        from keras.dtypes import DTypePolicy  
                        custom_objects['DTypePolicy'] = DTypePolicy
                    except ImportError:
                        logger.debug("DTypePolicy not found in mixed_precision or dtypes modules")
                
                logger.info("Registered custom objects (layers, compatible initializers) for model loading")
            except Exception as e:
                logger.warning(f"Could not register custom objects: {e}")
            
            # For .keras files (native Keras 3.x format)
            if self.model_path.endswith('.keras'):
                try:
                    # Try with custom_objects first
                    self.model = keras.models.load_model(self.model_path, compile=False, custom_objects=custom_objects)
                    logger.info(f"Successfully loaded .keras model from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Failed with custom_objects: {e}")
                    try:
                        # Try with safe_mode=False for .keras files too
                        self.model = keras.models.load_model(self.model_path, compile=False, safe_mode=False)
                        logger.info(f"Successfully loaded .keras model with safe_mode=False from {self.model_path}")
                    except Exception as e2:
                        logger.warning(f"Failed with safe_mode=False, trying default loading: {e2}")
                        self.model = keras.models.load_model(self.model_path, compile=False)
                        logger.info(f"Successfully loaded .keras model from {self.model_path}")
            else:
                # For .h5 files (legacy format), try safe_mode=False first
                try:
                    self.model = keras.models.load_model(self.model_path, compile=False, safe_mode=False, custom_objects=custom_objects)
                    logger.info(f"Successfully loaded .h5 model from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Failed with safe_mode=False, trying default loading: {e}")
                    self.model = keras.models.load_model(self.model_path, compile=False, custom_objects=custom_objects)
                    logger.info(f"Successfully loaded .h5 model from {self.model_path}")
        
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            raise RuntimeError(f"Model loading failed: {e}. Cannot proceed without a valid model.")

    def _load_savedmodel(self):
        """Load TensorFlow SavedModel - supports both GPU-trained and CPU-native models"""
        import tensorflow as tf

        logger.info(f"Loading SavedModel from directory: {self.model_path}")

        # Use direct TensorFlow inference (bypasses Keras/TFSMLayer GPU issues)
        try:
            loaded = tf.saved_model.load(self.model_path)

            if 'serving_default' not in loaded.signatures:
                raise ValueError(f"SavedModel at {self.model_path} does not have a 'serving_default' signature")

            infer = loaded.signatures['serving_default']

            # Get input/output specs
            input_spec = list(infer.structured_input_signature[1].values())[0]
            output_spec = list(infer.structured_outputs.values())[0]

            logger.info(f"SavedModel input shape: {input_spec.shape}, dtype: {input_spec.dtype}")
            logger.info(f"SavedModel output shape: {output_spec.shape}, dtype: {output_spec.dtype}")

            # Get the input key name for the signature
            input_key = list(infer.structured_input_signature[1].keys())[0]

            # Create wrapper class that mimics Keras model interface
            class SavedModelInferenceWrapper:
                def __init__(self, infer_func, inp_spec, out_spec, input_name):
                    self.infer_func = infer_func
                    self.input_spec = inp_spec
                    self.output_spec = out_spec
                    self.input_name = input_name
                    self.input_shape = (None,) + tuple(inp_spec.shape[1:])
                    self.output_shape = (None,) + tuple(out_spec.shape[1:])

                def predict(self, x, verbose=0):
                    """Make predictions using the SavedModel signature"""
                    # Convert input to TensorFlow tensor
                    x_tensor = tf.convert_to_tensor(x, dtype=self.input_spec.dtype)
                    # Call inference function with keyword argument
                    result = self.infer_func(**{self.input_name: x_tensor})
                    # Extract output (handle dict or direct tensor)
                    if isinstance(result, dict):
                        output = list(result.values())[0]
                    else:
                        output = result
                    # Convert back to numpy
                    return output.numpy()

            self.model = SavedModelInferenceWrapper(infer, input_spec, output_spec, input_key)

            logger.info(f"Successfully loaded SavedModel with direct inference wrapper (CPU-compatible):")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")

        except Exception as e:
            logger.error(f"Failed to load SavedModel: {e}")
            raise RuntimeError(f"SavedModel loading failed: {e}. Cannot proceed without a valid model.")

    def verify_model_architecture(self):
        """Verify the model has correct architecture after loading"""
        logger.info("\n" + "="*60)
        logger.info("VERIFYING MODEL ARCHITECTURE")
        logger.info("="*60)

        # Check for problematic layers
        has_global_pooling = False
        has_tanh_output = False
        output_shape = None

        if not hasattr(self.model, "layers"):
            logger.warning("Model has no .layers attribute (SavedModel wrapper). Skipping architecture verification.")
            return True


        for layer in self.model.layers:
            if 'GlobalAveragePooling' in str(type(layer)):
                has_global_pooling = True
                logger.error(f"ERROR: Found GlobalAveragePooling layer: {layer.name}")
                logger.error("This destroys temporal information and must be fixed in training!")

            if layer == self.model.layers[-1]:  # Last layer
                output_shape = layer.output_shape
                if hasattr(layer, 'activation'):
                    if 'tanh' in str(layer.activation):
                        has_tanh_output = True
                        logger.warning(f"WARNING: Output layer uses tanh activation")
                        logger.warning("This constrains outputs to [-1,1] and may limit performance")
                    else:
                        logger.info(f"Output layer activation: {layer.activation}")

        logger.info(f"Model output shape: {output_shape}")

        if has_global_pooling:
            logger.error("CRITICAL: Model has GlobalAveragePooling - predictions will be poor!")
            logger.error("Retrain the model with the architecture fixes.")

        if has_tanh_output:
            logger.warning("Model uses tanh output - may need adjustment in prediction scaling")

        return not has_global_pooling  # Return False if architecture is broken

    def test_model_temporal_sensitivity(self):
        """Test if model can distinguish between different temporal positions"""
        logger.info("\n" + "="*60)
        logger.info("TESTING TEMPORAL SENSITIVITY")
        logger.info("="*60)

        # Create test pattern
        test_value = 0.5

        # Test 1: Value at beginning (position 0 - critical for differencing)
        input1 = np.zeros((1, self.sequence_length, 1), dtype=np.float32)
        input1[0, 0, 0] = test_value

        # ----- ISSUE -----
        # With your SavedModel wrapper, predict() returns a numpy array shaped like (1, 96, 1) 
        # (your SavedModel output spec confirms this). So self.model.predict(...)[0, 0] is still an array 
        # (shape (1,) or (1,1) depending), not a scalar float — and formatting :.6f fails.
        
        
        # pred1 = self.model.predict(input1, verbose=0)[0, 0]
        
        
        # ----- END ISSUE -----

        # ----- PATCH -----
        # In test_model_temporal_sensitivity(), force scalar extraction 
        # which guarantees pred1/pred2/pred3 are scalars and your logging line works.

        pred1 = float(np.asarray(self.model.predict(input1, verbose=0)).reshape(-1)[0])
        
        # ----- END PATCH -----

        # Test 2: Value at middle
        input2 = np.zeros((1, self.sequence_length, 1), dtype=np.float32)
        input2[0, self.sequence_length//2, 0] = test_value

        # ----- ISSUE -----
        # same as above

        # pred2 = self.model.predict(input2, verbose=0)[0, 0]

        # ----- END ISSUE -----

        # ----- PATCH -----
        
        pred2 = float(np.asarray(self.model.predict(input2, verbose=0)).reshape(-1)[0])
        
        # ----- END PATCH -----

        # Test 3: Value at end (most recent)
        input3 = np.zeros((1, self.sequence_length, 1), dtype=np.float32)
        input3[0, -1, 0] = test_value

        # ----- ISSUE -----
        # same as above

        # pred3 = self.model.predict(input3, verbose=0)[0, 0]

        # ----- END ISSUE -----

        # ----- PATCH -----
        
        pred3 = float(np.asarray(self.model.predict(input3, verbose=0)).reshape(-1)[0])

        # ----- END PATCH -----

        logger.info(f"Value at position 0 (t-96) → prediction: {pred1:.6f}")
        logger.info(f"Value at position 48 (t-48) → prediction: {pred2:.6f}")
        logger.info(f"Value at position 95 (t-1) → prediction: {pred3:.6f}")

        # Calculate sensitivity
        max_diff = max(abs(pred1-pred2), abs(pred2-pred3), abs(pred1-pred3))

        if max_diff < 0.001:
            logger.error("ERROR: Model is NOT temporally sensitive!")
            logger.error("All positions produce similar predictions - GlobalAveragePooling issue!")
            return False
        else:
            logger.info(f"Good: Model shows temporal sensitivity (max diff: {max_diff:.6f})")
            return True

    # ----- START: ADDING NEW FUNCTION TO TEST WITH LOAD MODEL -----
    # Doing this to stop treating a many-to-many model as if it’s many-to-one and then recursing.
    # what this function does:
    # - take the (1,96,1) output
	# - denormalize all 96 predicted deltas
	# - add each delta to its base value (t-96 baseline) → get 96 forecasts
	# - no recursion
    def _perform_seq2seq_forecast(self, history_window_norm, start_index):
        """
        Non-recursive seq2seq forecast:
        - one model call returns (96,1) normalized diffs
        - denormalize to diffs
        - inverse difference using base_value = original_values[(start_index + k) - daily_lag]
        """
        if self.dataset_kind != "load":
            raise RuntimeError("_perform_seq2seq_forecast is intended only for load data.")
        x = np.asarray(history_window_norm, dtype=np.float32).reshape(1, self.sequence_length, 1)

        pred = self.model.predict(x, verbose=0)
        pred_arr = np.asarray(pred).reshape(self.sequence_length)  # 96 normalized diffs


        # DEBUG: verify what the model is predicting (seasonal diffs vs step-to-step deltas)
        if start_index >= self.daily_lag and start_index + self.sequence_length <= len(self.original_values):
            true_seasonal_diff = (
                self.original_values[start_index:start_index+self.sequence_length]
                - self.original_values[start_index-self.daily_lag:start_index-self.daily_lag+self.sequence_length]
            )
            dbg_stats("true_seasonal_diff_96", true_seasonal_diff)
            dbg_stats("pred_norm_change_96", pred_arr)


        forecast_trace = []

        a = self.feature_range_min
        b = self.feature_range_max

        for k in range(self.sequence_length):
            # normalized -> differenced scale
            pred_norm_change = float(pred_arr[k])
            pred_change = self.data_min + (pred_norm_change - a) * (self.data_max - self.data_min) / (b - a)

            # inverse differencing base
            base_idx = (start_index + k) - self.daily_lag
            base_value = float(self.original_values[base_idx]) if 0 <= base_idx < len(self.original_values) else float('nan')

            preclamp = base_value + pred_change if np.isfinite(base_value) else float('nan')
            final_forecast = 0.0 if (np.isfinite(preclamp) and preclamp < 0) else float(preclamp)

            forecast_trace.append({
                'pred_norm_change': float(pred_norm_change),
                'predicted_change': float(pred_change),
                'final_forecast': float(final_forecast),
                'base_value': float(base_value),
            })

        return forecast_trace

    # ----- END: ADDING NEW FUNCTION TO TEST WITH LOAD MODEL -----

    def _perform_recursive_forecast(self, initial_history_window_norm, start_index):
        """Perform auto-regressive forecast using the loaded Keras model"""
        logger.debug(f"Starting auto-regressive forecast for window starting at index {start_index}...")

        window_norm = np.asarray(initial_history_window_norm, dtype=float).reshape(-1)
        if window_norm.shape[0] != self.sequence_length:
            logger.warning(f"Initial normalized window length {window_norm.shape[0]} != sequence_length {self.sequence_length}")

        forecast_trace = []

        # Track prediction quality
        prediction_values = []  # Track all predictions
        prediction_changes = []  # Track step-to-step changes

        # PRODUCTION: Base values always come from actual historical data (96 steps ago)
        # No need for prediction-based base value buffer since we always have actuals
        base_value_buffer = []

        for i in range(self.sequence_length):
            # Calculate the position we're predicting
            prediction_position = start_index + i
            # Base value comes from 96 steps before the prediction position
            base_value_index = prediction_position - self.daily_lag

            if 0 <= base_value_index < len(self.original_values):
                base_value_buffer.append(float(self.original_values[base_value_index]))
            else:
                # Edge case: use last available value if beyond data range
                base_value_buffer.append(base_value_buffer[-1] if base_value_buffer else 0.0)

        # Track the raw window for reporting (starts with base values)
        current_raw_window = base_value_buffer.copy()

        for step in range(self.sequence_length):
            # Current input for this step
            input_for_this_step = window_norm.copy()

            # Predict next normalized differenced change using Keras model
            model_input = input_for_this_step.reshape(1, self.sequence_length, 1)


            # ----- LOGGING START-----
            if step == 0 and start_index < (self.sequence_length + 5):  # only for very early windows to avoid spam
                print(f"[DBG] model_input shape={model_input.shape} dtype={model_input.dtype} "
                    f"nan={np.isnan(model_input).sum()} inf={np.isinf(model_input).sum()} "
                    f"min={np.nanmin(model_input):.4f} max={np.nanmax(model_input):.4f}")
            # ----- LOGGING END-----

            pred = self.model.predict(model_input, verbose=0)

            # SANITY CHECK: This avoids silently using the wrong element if the model is seq2seq.
            pred_arr = np.asarray(pred)
            if pred_arr.ndim > 2 and pred_arr.shape[1] == self.sequence_length:
                # if model outputs a full sequence, take the last step (most recent)
                original_pred = float(pred_arr.reshape(-1)[-1])
            else:
                original_pred = float(pred_arr.reshape(-1)[0])
            # SANITY CHECK END

            # ----- LOGGING START -----
            pred_arr = np.asarray(pred)
            if step == 0 and start_index < (self.sequence_length + 5):
                print(f"[DBG] raw_model_output shape={pred_arr.shape} "
                    f"min={pred_arr.min():.4f} p50={np.median(pred_arr):.4f} max={pred_arr.max():.4f}")
            # ----- LOGGING END -----

            # original_pred = float(np.array(pred).reshape(-1)[0])

            # Allow wider range for linear activation models
            EXPECTED_OUTPUT_RANGE = 3.0  # Match the soft clipping in training

            # Only warn if significantly outside expected range
            if abs(original_pred) > EXPECTED_OUTPUT_RANGE * 1.1:
                logger.warning(f"Model output {original_pred:.4f} exceeds expected range [-{EXPECTED_OUTPUT_RANGE}, {EXPECTED_OUTPUT_RANGE}]")
                pred_norm_change = np.clip(original_pred, -EXPECTED_OUTPUT_RANGE, EXPECTED_OUTPUT_RANGE)
            else:
                pred_norm_change = original_pred  # Use as-is if within range

            # De-normalize the predicted change from [-1, 1] back to differenced scale
            # PRODUCTION ORDER: normalize → predict → denormalize → inverse difference
            if self.data_max is not None and self.data_min is not None:
                a = self.feature_range_min
                b = self.feature_range_max
                pred_change = self.data_min + (pred_norm_change - a) * (self.data_max - self.data_min) / (b - a)
            else:
                pred_change = float(pred_norm_change)

            # Use base value from actual historical data (96 steps ago)
            base_value = base_value_buffer[step]

            # Inverse differencing: Add predicted change to base value (from 96 steps ago)
            # This converts from differenced space back to absolute load values
            final_forecast = base_value + float(pred_change)
            if final_forecast < 0:
                final_forecast = 0.0  # Clamp to prevent negative load values
            
            if step < 3 and start_index < (self.sequence_length + 5):
                print(f"[DBG] step={step} base_value={base_value:.3f} pred_norm_change={pred_norm_change:.3f} " 
                      f"pred_change={pred_change:.3f} final_forecast={final_forecast:.3f}")

            # Track prediction quality
            prediction_values.append(final_forecast)
            if len(prediction_values) > 1:
                change = final_forecast - prediction_values[-2]
                prediction_changes.append(change)

            # Prepare input window in differenced space for reporting
            if self.data_max is not None and self.data_min is not None:
                a = self.feature_range_min
                b = self.feature_range_max
                input_window_diff = self.data_min + (input_for_this_step - a) * (self.data_max - self.data_min) / (b - a)
            else:
                input_window_diff = input_for_this_step.copy()

            forecast_trace.append({
                'input_window_diff': np.asarray(input_window_diff).reshape(-1),
                'input_window_raw': current_raw_window.copy(),
                'pred_norm_change': float(pred_norm_change),
                'predicted_change': float(pred_change),
                'final_forecast': float(final_forecast),
                'base_value': float(base_value)
            })

            # Update the normalized history window for next step (recursive)
            window_norm = np.append(window_norm[1:], pred_norm_change)

            # Update the raw window for next step (shift and append prediction)
            if step < self.sequence_length - 1:
                current_raw_window = current_raw_window[1:] + [final_forecast]

        # Check for prediction collapse
        if len(prediction_values) >= 24:
            last_24_std = np.std(prediction_values[-24:])
            if last_24_std < 0.01:
                logger.warning(f"WARNING: Predictions collapsed to near-constant! Std of last 24: {last_24_std:.6f}")

            # Check if predictions are stuck
            if len(set(np.round(prediction_values[-10:], 4))) == 1:
                logger.error("ERROR: Model producing identical predictions - stuck in local minimum!")

        logger.debug("Auto-regressive forecast complete.")
        return forecast_trace

    def run_simulation(self):
        logger.info("Starting implementation simulation (production-style sliding window forecast)...")
        all_results = []

        # Read the controllableLoadForecast_0 column (index 4) for load data
        values = self.test_data.iloc[:, 4].values  # Differenced values
        n = len(values)
        window_size = self.sequence_length

        # Calculate total windows: slide by 1 timestep starting from window_size
        # We need window_size points before we can start forecasting
        # Maximum start position: we need enough data to forecast sequence_length ahead
        original_data_size = len(self.original_values) if self.original_values is not None else n + self.daily_lag
        max_prediction_end = original_data_size - 1
        max_start_position = min(n - 1, len(self.diff_orig_idx) - 1, max_prediction_end - self.sequence_length + 1)

        # Production: slide by 1 timestep (15 mins), not by sequence_length
        total_windows = max_start_position - window_size + 1

        if total_windows <= 0:
            logger.warning("Not enough data for even a single rolling window.")
            return None

        log_interval = max(1, total_windows // 10)
        logger.info(f"Total rolling windows to process: {total_windows} (sliding by 1 timestep)")

        # Slide window by 1 timestep (production behavior)
        for idx, t in enumerate(range(window_size, max_start_position + 1)):

            # Map the differenced-row position `t` back to the original-series index `orig_t`.
            # After 0→NaN and diff()+dropna(), the differenced array indices no longer align with original_values,
            # so using `t` directly causes a fixed/variable offset in Actual vs Pred.
            # We therefore use `orig_t` everywhere we index original_values or define absolute-time alignment:
            # - actuals_start
            # - forecast start_index passed into _perform_seq2seq_forecast / _perform_recursive_forecast
            # - raw_input_window indexing
            # - ALIGN_AUDIT ts/base/actual indices
            # - stored results['timestamp'] used later by _generate_report()
            orig_t = int(self.diff_orig_idx[t])

            if idx == 0:
                logger.info(f"Processing first window at index {t}")
            if (idx + 1) % log_interval == 0 or (idx + 1) == total_windows:
                logger.info(f"Progress: {idx + 1}/{total_windows} windows ({((idx + 1) / total_windows) * 100:.1f}%)")

            # PRODUCTION: Always use actual historical differenced data for input window
            # Window slides by 1: [t-96 to t-1] → [t-95 to t] → [t-94 to t+1] etc.
            history_window_diff = values[t - window_size:t]

            a = self.feature_range_min
            b = self.feature_range_max

            if idx < 3:
                dbg_stats("history_window_diff_pre_norm", history_window_diff)
                den = (self.data_max - self.data_min)
                print(f"[DBG] norm params: data_min={self.data_min} data_max={self.data_max} den={den} range=[{a},{b}]")
                print(f"[DBG] history_window_diff outside min/max: below={(history_window_diff < self.data_min).mean():.3%} above={(history_window_diff > self.data_max).mean():.3%}")


            # Normalize the input window (always from actual historical data)
            if self.data_max is not None and self.data_min is not None:
                a = self.feature_range_min
                b = self.feature_range_max
                initial_history_norm = a + (history_window_diff - self.data_min) * (b - a) / (self.data_max - self.data_min)
            else:
                initial_history_norm = history_window_diff.copy()
            if idx < 3:
                dbg_stats("initial_history_norm_post_norm", initial_history_norm)
                print(f"[DBG] norm out of [{a},{b}]: below={(initial_history_norm < a).mean():.3%} above={(initial_history_norm > b).mean():.3%}")


            # Get absolute actuals (original series) aligned to predictions
            # When we're at position t in differenced data, we predict for position t in original data
            # actuals_start = t
            actuals_start = orig_t
            actuals_end = min(actuals_start + window_size, len(self.original_values))
            actuals = self.original_values[actuals_start:actuals_end]
            # Pad with NaN if we predict beyond available data
            if len(actuals) < window_size:
                actuals = list(actuals) + [float('nan')] * (window_size - len(actuals))

            # Forecast using normalized auto-regressive loop (unchanged)
            # forecast_trace = self._perform_recursive_forecast(initial_history_norm, t)
            # ----- START PATCH: make sure the new function is triggered only for load data -----
            if self.dataset_kind == "load":
                # forecast_trace = self._perform_seq2seq_forecast(initial_history_norm, t)
                forecast_trace = self._perform_seq2seq_forecast(initial_history_norm, orig_t)
            else:
                forecast_trace = self._perform_recursive_forecast(initial_history_norm, orig_t)
            # ----- END PATCH -----
            final_predictions = [step['final_forecast'] for step in forecast_trace]

            # Construct the raw input window from actual historical values
            raw_input_window = []
            for pos in range(window_size):
                # input_position = t - window_size + pos
                input_position = orig_t - window_size + pos
                if 0 <= input_position < len(self.original_values):
                    raw_input_window.append(self.original_values[input_position])
                else:
                    raw_input_window.append(0.0)
            
            # ----- START: Confirm base value indexing JUST FOR ONE WINDOW -----
            # ===== ALIGNMENT AUDIT: run only for Rolling Window 0 =====
            if idx == 0:
                # ts = t
                ts = orig_t
                lag = self.daily_lag
                L = self.sequence_length

                print("\n" + "="*80)
                print("[ALIGN_AUDIT] Rolling Window 0")
                print(f"[ALIGN_AUDIT] ts(t)={ts}  lag={lag}  seq_len={L}")
                print(f"[ALIGN_AUDIT] input window idx range: [{ts-L} .. {ts-1}]")
                print(f"[ALIGN_AUDIT] target idx range (Actual/Pred): [{ts} .. {ts+L-1}]")
                print(f"[ALIGN_AUDIT] base idx range: [{ts-lag} .. {ts+L-1-lag}]")
                print("="*80)

                print("k | input_idx actual_idx base_idx | Input     Actual    Base      "
                    "Pred      PredChg    PreClamp   Clamped")
                print("-"*80)

                for k in range(0, 21):
                    input_idx = ts - L + k
                    actual_idx = ts + k
                    base_idx = ts + k - lag

                    # inp = float(self.original_values[input_idx]) if 0 <= input_idx < len(self.original_values) else float('nan')
                    inp = float(raw_input_window[k])
                    act = float(self.original_values[actual_idx]) if 0 <= actual_idx < len(self.original_values) else float('nan')
                    base = float(self.original_values[base_idx]) if 0 <= base_idx < len(self.original_values) else float('nan')

                    pred = float(forecast_trace[k]['final_forecast'])
                    pred_chg = float(forecast_trace[k]['predicted_change'])  # after denorm, before adding base
                    # preclamp = base + pred_chg
                    # clamped = preclamp < 0
                    preclamp = base + pred_chg if np.isfinite(base) else float('nan')
                    clamped = (np.isfinite(preclamp) and preclamp < 0)

                    print(f"{k:2d} | {input_idx:8d} {actual_idx:9d} {base_idx:8d} | "
                        f"{inp:8.3f} {act:8.3f} {base:8.3f} "
                        f"{pred:8.3f} {pred_chg:9.3f} {preclamp:9.3f} {str(clamped):>7}")
            # ----- End: Confirm base value indexing JUST FOR ONE WINDOW -----


            all_results.append({
                # 'timestamp': t,  # Position in original data where predictions start
                'timestamp': orig_t,
                'actuals': actuals,
                'predictions': final_predictions,
                'trace': forecast_trace,
                'raw_input_window': raw_input_window,  # 96-length window of actual absolute values
                'window_index': idx  # Track which window this is
            })

            if idx == total_windows - 1:
                logger.info(f"Finished processing last window at index {t}")

        logger.info("Simulation complete. Generating report...")
        return self._generate_report(all_results)

    def _generate_report(self, results):
        """Generate performance reports"""
        logger.info("Generating performance reports...")
        
        # Prepare data for metrics calculation
        all_actuals = []
        all_preds = []
        horizon_errors = {4: [], 12: [], 24: [], 48: [], 96: []}  # 1h, 3h, 6h, 12h, 24h
        
        for entry in results:
            ts = entry['timestamp']
            preds = entry['predictions']

            # Align original actuals to this window
            aligned_actuals = []
            for step in range(min(len(preds), self.sequence_length)):
                original_idx = ts + step
                if 0 <= original_idx < len(self.original_values):
                    aligned_actuals.append(self.original_values[original_idx])
                else:
                    aligned_actuals.append(np.nan)

            # Calculate horizon errors
            for horizon in [4, 12, 24, 48, 96]:
                if len(aligned_actuals) >= horizon and len(preds) >= horizon:
                    actuals_horizon = np.array(aligned_actuals[:horizon])
                    preds_horizon = np.array(preds[:horizon])
                    # Filter out NaN values for MAE calculation
                    valid_mask = ~np.isnan(actuals_horizon)
                    if np.sum(valid_mask) > 0:  # Only calculate if we have valid actual values
                        window_mae = mean_absolute_error(
                            actuals_horizon[valid_mask], preds_horizon[valid_mask]
                        )
                        horizon_errors[horizon].append(window_mae)

            # Aggregate for overall metrics
            for i in range(min(len(aligned_actuals), len(preds))):
                a = aligned_actuals[i]
                p = preds[i]
                # Only add if actual value is not NaN
                if not np.isnan(a):
                    all_actuals.append(a)
                    all_preds.append(p)
        
        # Calculate overall metrics (only if we have valid data)
        if len(all_actuals) > 0:
            mae = mean_absolute_error(all_actuals, all_preds)
            rmse = float(np.sqrt(mean_squared_error(all_actuals, all_preds)))
        else:
            mae = float('nan')
            rmse = float('nan')
        logger.info(f'Overall MAE: {mae:.4f}')
        logger.info(f'Overall RMSE: {rmse:.4f}')
        
        # Log horizon errors
        logger.info('Average MAE by forecast horizon:')
        for horizon, errors in horizon_errors.items():
            if errors:
                avg_mae = np.mean(errors)
                logger.info(f'  First {horizon} steps ("{horizon//4}" hours): MAE = {avg_mae:.4f}')
            else:
                logger.info(f'  First {horizon} steps ("{horizon//4}" hours): MAE = N/A')
        
        # Generate simplified long-format report
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        simplified_rows = []

        # Create one row per actual data point
        for i, actual_val in enumerate(self.original_values):
            prediction_val = None

            # FIXING THE FOLLOWING 
            # Because we break, we are taking the earliest window that contains i, not the latest. With sliding windows, 
            # each timestamp is covered by many windows; typically we want:
            # the prediction produced at the most recent origin for that timestamp
            # i.e. “the window whose timestamp is closest to i but not greater than i”
            # Fix: iterate reversed so you take the most recent covering window.
            # for entry in results:
            for entry in reversed(results):
                if entry['timestamp'] <= i < entry['timestamp'] + len(entry['predictions']):
                    pred_idx = i - entry['timestamp']
                    if 0 <= pred_idx < len(entry['predictions']):
                        prediction_val = entry['predictions'][pred_idx]
                        break

            if prediction_val is None:
                prediction_val = actual_val  # Fallback to actual value

            simplified_rows.append({
                'timestamp': i,
                'actual': actual_val,
                'prediction': prediction_val,
                'error': prediction_val - actual_val
            })

        df_long = pd.DataFrame(simplified_rows)
        long_csv = f'reports/performance_report_{model_name}.csv'
        df_long.to_csv(long_csv, index=False)
        logger.info(f'Simplified long-format report saved to {long_csv} with {len(simplified_rows)} rows')

        # Generate wide-format report
        self._generate_wide_format_report(results, model_name)
        
        # Return metrics
        return {
            'model_name': model_name,
            'mae_1hr': np.mean(horizon_errors[4]) if horizon_errors[4] else np.nan,
            'mae_3hr': np.mean(horizon_errors[12]) if horizon_errors[12] else np.nan,
            'mae_6hr': np.mean(horizon_errors[24]) if horizon_errors[24] else np.nan,
            'mae_12hr': np.mean(horizon_errors[48]) if horizon_errors[48] else np.nan,
            'mae_overall': mae
        }

    def _generate_wide_format_report(self, results, model_name):
        """Generate detailed wide-format report"""
        logger.info("Generating wide-format report...")

        all_windows_dfs = []

        for window_idx, entry in enumerate(results):
            if 'trace' not in entry:
                logger.warning(f"Entry {window_idx} missing trace; skipping.")
                continue

            trace = entry['trace']
            row_labels = [f"t+{i}" for i in range(self.sequence_length)]
            metric_labels = ["MAE_Overall", "MAE_3h", "MAE_6h", "MAE_12h", "MAE_24h"]
            full_index = row_labels + metric_labels

            # Simplified: only 4 columns per window (Input, Actual, Predicted, Error)
            column_headers = ["Input", "Actual", "Predicted", "Error"]
            df_window = pd.DataFrame(index=full_index, columns=column_headers, dtype=object)

            actuals_absolute = entry['actuals']
            predictions_absolute = entry['predictions']

            # Get the raw input window (initial actual values before forecasting)
            input_window_raw = np.array(entry['raw_input_window'])

            # Populate all 96 forecast steps
            for t in range(self.sequence_length):
                step_data = trace[t]

                # Input: raw actual value at this position in the input window
                df_window.loc[f"t+{t}", "Input"] = input_window_raw[t] if t < len(input_window_raw) else np.nan

                # Actual: ground truth value
                if t < len(actuals_absolute):
                    df_window.loc[f"t+{t}", "Actual"] = actuals_absolute[t]
                    df_window.loc[f"t+{t}", "Error"] = step_data['final_forecast'] - actuals_absolute[t]
                else:
                    df_window.loc[f"t+{t}", "Actual"] = np.nan
                    df_window.loc[f"t+{t}", "Error"] = np.nan

                # Predicted: model forecast
                df_window.loc[f"t+{t}", "Predicted"] = step_data['final_forecast']

            # Helper function to calculate MAE with NaN handling
            def safe_mae(actuals, preds):
                actuals_arr = np.array(actuals)
                preds_arr = np.array(preds)
                valid_mask = ~np.isnan(actuals_arr)
                if np.sum(valid_mask) > 0:
                    return mean_absolute_error(actuals_arr[valid_mask], preds_arr[valid_mask])
                return np.nan
            
            # Calculate MAE metrics for this window
            df_window.loc["MAE_Overall", "Predicted"] = safe_mae(actuals_absolute, predictions_absolute)
            if len(predictions_absolute) >= 12:
                df_window.loc["MAE_3h", "Predicted"] = safe_mae(actuals_absolute[:12], predictions_absolute[:12])
            if len(predictions_absolute) >= 24:
                df_window.loc["MAE_6h", "Predicted"] = safe_mae(actuals_absolute[:24], predictions_absolute[:24])
            if len(predictions_absolute) >= 48:
                df_window.loc["MAE_12h", "Predicted"] = safe_mae(actuals_absolute[:48], predictions_absolute[:48])
            if len(predictions_absolute) >= self.sequence_length:
                df_window.loc["MAE_24h", "Predicted"] = safe_mae(actuals_absolute[:self.sequence_length], predictions_absolute[:self.sequence_length])

            # Add header for rolling window
            df_window.columns = pd.MultiIndex.from_product([[f"Rolling Window {window_idx}"], df_window.columns])
            all_windows_dfs.append(df_window)

        if not all_windows_dfs:
            logger.warning("No data available to generate the wide-format report.")
            return None

        # Concatenate all rolling window DataFrames
        final_df = pd.concat(all_windows_dfs, axis=1)

        output_file = f"reports/performance_report_{model_name}_wide.csv"
        final_df.to_csv(output_file)
        logger.info(f"Wide-format report saved to {output_file}")

        return output_file

def run_all_models_implementation():
    """Run implementation for all available models in the models directory"""
    logger.info("Starting implementation for all models in models directory...")
    
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
    
    # Discover model files (.h5 and .keras)
    model_files = []
    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        if os.path.isfile(filepath) and (filename.endswith('.h5') or filename.endswith('.keras')):
                model_files.append((filename, filepath))
    
    # Discover CSV files
    csv_files = []
    for filename in os.listdir(impl_dir):
        filepath = os.path.join(impl_dir, filename)
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            csv_files.append((filename, filepath))
    
    logger.info(f"Found {len(model_files)} model files: {[f[0] for f in model_files]}")
    logger.info(f"Found {len(csv_files)} CSV files: {[f[0] for f in csv_files]}")
    
    if not model_files:
        logger.warning("No .h5 or .keras model files found in models directory!")
        return
    
    if not csv_files:
        logger.warning("No CSV files found in implementation directory!")
        return
    
    all_results = []
    
    # Run implementation for each model with each CSV file
    for model_filename, model_path in model_files:
        for csv_filename, csv_path in csv_files:
            try:
                logger.info(f"Running implementation for model: {model_filename} with data: {csv_filename}")
                
                impl = implementation(model_path, csv_path, 'config.ini')
                impl.load_dependencies()
                
                metrics = impl.run_simulation()
                if metrics:
                    metrics['model_name'] = f"{os.path.splitext(model_filename)[0]}_{os.path.splitext(csv_filename)[0]}"
                    all_results.append(metrics)
                    logger.info(f"Completed implementation for {model_filename} with {csv_filename}")
                
            except Exception as e:
                logger.error(f"Error running implementation for model {model_filename} with data {csv_filename}: {str(e)}")
                continue
    
    # Generate comparative summary
    if all_results:
        generate_comparative_summary(all_results)
    else:
        logger.warning("No successful model implementations to compare.")

def generate_comparative_summary(all_results):
    """Generate comparative summary CSV"""
    logger.info("Generating comparative summary CSV...")
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'model': result['model_name'],
            'MAE 1HR': result['mae_1hr'],
            'MAE 3HR': result['mae_3hr'],
            'MAE 6HR': result['mae_6hr'],
            'MAE 12HR': result['mae_12hr'],
            'MAE Overall': result['mae_overall']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    output_file = "reports/comparative_summary.csv"
    df_summary.to_csv(output_file, index=False)
    logger.info(f"Comparative summary saved to {output_file}")
    
    logger.info("Comparative Summary:")
    logger.info(df_summary.to_string(index=False))
    
    return output_file