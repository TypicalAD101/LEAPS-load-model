# #!/usr/bin/env python3
# """
# export_cpu_savedmodel.py

# CPU-safe SavedModel export:
# - Loads a portable .keras model
# - Forces CPU-only tracing (prevents CuDNN ops like CudnnRNNV3)
# - Exports a TF SavedModel that runs on CPU-only TensorFlow
# """

# import os

# # IMPORTANT: must be set BEFORE importing tensorflow/keras
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import argparse
# import numpy as np
# import tensorflow as tf


# def set_float32_policy():
#     # Make sure we don’t export a model locked to float16 kernels.
#     try:
#         tf.keras.mixed_precision.set_global_policy("float32")
#     except Exception:
#         pass

# @tf.keras.utils.register_keras_serializable(package="Custom")
# class ClipByValue(tf.keras.layers.Layer):
#     def __init__(self, min_value, max_value, **kwargs):
#         super().__init__(**kwargs)
#         self.min_value = float(min_value)
#         self.max_value = float(max_value)
#     def call(self, x):
#         return tf.clip_by_value(x, self.min_value, self.max_value)
#     def get_config(self):
#         cfg = super().get_config()
#         cfg.update({"min_value": self.min_value, "max_value": self.max_value})
#         return cfg


# def export_cpu_savedmodel(keras_path: str, export_dir: str, seq_len: int = 96):
#     set_float32_policy()

#     if not os.path.isfile(keras_path):
#         raise FileNotFoundError(f"Keras model not found: {keras_path}")

#     os.makedirs(export_dir, exist_ok=True)

#     # Load .keras model
#     model = tf.keras.models.load_model(keras_path, compile=False, safe_mode=False, custom_objects={"ClipByValue": ClipByValue})

#     # Build a CPU-safe serving function with float32 signature
#     @tf.function(
#         input_signature=[
#             tf.TensorSpec(shape=[None, seq_len, 1], dtype=tf.float32, name="keras_tensor")
#         ]
#     )
#     def serving_fn(x):
#         # ensure float32
#         x = tf.cast(x, tf.float32)
#         y = model(x, training=False)
#         y = tf.cast(y, tf.float32)
#         return {"output_0": y}

#     tf.saved_model.save(model, export_dir, signatures={"serving_default": serving_fn})

#     # Quick sanity check: print devices + signature keys
#     loaded = tf.saved_model.load(export_dir)
#     sig = loaded.signatures["serving_default"]
#     print("CPU-safe export complete.")
#     print("Devices visible:", tf.config.list_physical_devices())
#     print("Signature input keys:", list(sig.structured_input_signature[1].keys()))
#     print("Signature output keys:", list(sig.structured_outputs.keys()))
#     print("Export dir:", export_dir)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--keras_path", required=True, help="Path to .keras file")
#     ap.add_argument("--export_dir", required=True, help="Output directory for CPU-safe SavedModel")
#     ap.add_argument("--seq_len", type=int, default=96, help="Sequence length (default=96)")
#     args = ap.parse_args()

#     export_cpu_savedmodel(args.keras_path, args.export_dir, args.seq_len)


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
export_cpu_savedmodel.py

CPU-safe SavedModel export:
- Loads a portable .keras model
- Forces CPU-only tracing (prevents CuDNN ops like CudnnRNNV3)
- Exports a TF SavedModel that runs on CPU-only TensorFlow
"""

import os

# IMPORTANT: must be set BEFORE importing tensorflow/keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import tensorflow as tf

# =============================================================================
# 1. CUSTOM LAYER REGISTRATION
# =============================================================================
# The decorator must match exactly how it was saved in the .keras file.
# Removing 'package="Custom"' because the error log showed Keras was 
# looking for the plain registered name 'ClipByValue'.
@tf.keras.utils.register_keras_serializable()
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

def set_float32_policy():
    """Ensure the export doesn't use float16/mixed_precision kernels."""
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass

def export_cpu_savedmodel(keras_path: str, export_dir: str, seq_len: int = 96):
    set_float32_policy()

    if not os.path.isfile(keras_path):
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    # Remove the manual os.makedirs(export_dir) here because model.export 
    # handles directory creation and needs a clean or non-existent path.

    # 1. LOAD MODEL WITH CUSTOM OBJECTS
    print(f"Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(
        keras_path, 
        compile=False, 
        safe_mode=False, 
        custom_objects={"ClipByValue": ClipByValue}
    )

    # 2. Use the Keras 3 export method
    # This automatically handles the serving signature and bypasses the _DictWrapper bug.
    # We use this INSTEAD of tf.saved_model.save to avoid serialization errors.
    print(f"Exporting via Keras 3 export to: {export_dir}")
    model.export(export_dir)

    # 3. Quick sanity check
    try:
        loaded = tf.saved_model.load(export_dir)
        sig = loaded.signatures["serving_default"]
        print("====================================================")
        print("CPU-safe export complete.")
        print("Devices visible:", tf.config.list_physical_devices())
        print("Signature output keys:", list(sig.structured_outputs.keys()))
        print("====================================================")
    except Exception as e:
        print(f"Warning: Export finished, but sanity check failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras_path", required=True, help="Path to .keras file")
    ap.add_argument("--export_dir", required=True, help="Output directory for CPU-safe SavedModel")
    ap.add_argument("--seq_len", type=int, default=96, help="Sequence length (default=96)")
    args = ap.parse_args()

    export_cpu_savedmodel(args.keras_path, args.export_dir, args.seq_len)

if __name__ == "__main__":
    main()