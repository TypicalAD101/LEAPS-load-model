import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from Training.base_trainer import BaseTrainer
from utils.logger import get_logger

class keras_model_train(BaseTrainer):
    """
    Corrected and optimized trainer for a Keras LSTM model.
    Implements a robust custom training loop with scheduled sampling,
    designed for the recursive forecasting needs of the simulation module.
    """
    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def _build_model(self, seq_length=96, lstm_units_1=128, lstm_units_2=64, dense_units=32):
        """
        Builds the simple, many-to-one LSTM model required by the forecast_module.
        """
        self.logger.info("Building stateless, single-step LSTM model.")
        self.model = Sequential([
            Input(shape=(seq_length, 1)),
            LSTM(lstm_units_1, return_sequences=True),
            Dropout(0.3),
            LSTM(lstm_units_2, return_sequences=False), # returns only the last output
            Dropout(0.3),
            Dense(dense_units, activation='relu'),
            Dropout(0.2),
            Dense(1) # Linear activation for regression output
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.logger.info("Model built successfully.")

    def _create_dataset_for_simulation(self, data, seq_length=96, forecast_horizon=96):
        """
        Creates the dataset where X is history and y is the future sequence.
        This is required for the custom training loop to calculate loss correctly.
        """
        # Ensure data is 1D before windowing
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        self.logger.info(f"Creating training windows: X_shape=({seq_length}, 1), y_shape=({forecast_horizon}, 1)")
        X, y = [], []
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            x_window = data[i:(i + seq_length)]
            # Always ensure x_window is (seq_length, 1)
            if x_window.ndim == 1:
                x_window = x_window[:, np.newaxis]
            X.append(x_window)
            y.append(data[(i + seq_length):(i + seq_length + forecast_horizon)])
        X = np.array(X, dtype=np.float32)  # (num_samples, seq_length, 1)
        y = np.array(y, dtype=np.float32)
        if y.ndim == 2:
            y = y[..., np.newaxis]  # (num_samples, forecast_horizon, 1)
        return X, y

    @tf.function
    def _perform_train_step(self, x_batch, y_batch, sampling_probability):
        """
        Performs a single training step for one batch using a recursive forecast simulation.
        Compiled with @tf.function for high performance.
        """
        with tf.GradientTape() as tape:
            batch_forecasts = tf.TensorArray(tf.float32, size=tf.shape(x_batch)[0])
            
            # Loop through each sample in the batch
            for i in tf.range(tf.shape(x_batch)[0]):
                history_window = x_batch[i]
                history_window = tf.cast(history_window, tf.float32)  # Ensure float32 for concat
                y_true_sequence = y_batch[i]
                
                single_forecast = tf.TensorArray(tf.float32, size=tf.shape(y_true_sequence)[0])
                
                # Perform the 96-step recursive forecast for this one sample
                forecast_horizon = tf.shape(y_true_sequence)[0]
                for t in tf.range(forecast_horizon):
                    # tf.print('history_window shape before model input:', tf.shape(history_window))
                    # Only expand dims for model input
                    model_input = tf.expand_dims(history_window, 0)  # (1, 96, 1)
                    next_step_pred = self.model(model_input, training=True)[0]
                    single_forecast = single_forecast.write(t, next_step_pred)

                    # Scheduled Sampling: Decide what to use for the next input
                    use_true_value = tf.random.uniform(()) < sampling_probability
                    next_input_step = tf.cond(
                        use_true_value,
                        lambda: tf.cast(y_true_sequence[t], tf.float32),
                        lambda: next_step_pred[0]
                    )
                    # Reshape next_input_step to (1, 1) before concatenation
                    next_input_step_reshaped = tf.reshape(next_input_step, (1, 1))
                    # history_window[1:] is (95, 1), next_input_step_reshaped is (1, 1)
                    history_window = tf.concat([history_window[1:], next_input_step_reshaped], axis=0)
                    history_window.set_shape([96, 1])
                
                batch_forecasts = batch_forecasts.write(i, single_forecast.stack())

            # Calculate loss across all the 96-step forecasts in the batch
            final_forecasts = batch_forecasts.stack()
            loss = self.loss_fn(y_batch, final_forecasts)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, X_train, y_train, seq_length=96, model_save_path: str = None, epochs=50, batch_size=32, **kwargs):
        """
        Main training method.
        """
        # 1. Build the model
        self._build_model(seq_length=seq_length)

        # 2. Use X_train and y_train directly (already windowed)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size)
        
        # 3. Run the custom training loop
        self.logger.info("Starting custom training loop...")
        for epoch in range(epochs):
            # Scheduled sampling probability decays linearly from 1.0 to 0.1
            sampling_prob = 1.0 - 0.9 * (epoch / max(epochs - 1, 1))
            
            epoch_loss_avg = tf.keras.metrics.Mean()

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss = self._perform_train_step(x_batch, y_batch, tf.constant(sampling_prob, dtype=tf.float32))
                epoch_loss_avg.update_state(loss)
                print(f"\rEpoch {epoch+1}/{epochs}, Batch {step+1}, Loss: {loss:.6f}", end="")

            self.logger.info(f"Epoch {epoch+1} - Avg Loss: {epoch_loss_avg.result():.6f} - Sampling Prob: {sampling_prob:.3f}")
            epoch_loss_avg.reset_states()
        
        # 4. Save the TRAINED model
        if model_save_path is None:
            model_save_path = 'model_keras.h5'
        self.model.save(model_save_path)
        self.logger.info(f'Trained model saved to {model_save_path}')

