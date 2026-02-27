from Training.base_trainer import BaseTrainer
import tensorflow as tf
from hyperopt import hp as hpo
import keras_tuner as kt
from utils.logger import get_logger

class keras_tuner_model(BaseTrainer):
    """
    Trainer for Keras models using Keras Tuner for hyperparameter optimization.
    """
    def __init__(self):
        self.logger = get_logger()

    def run(self, window_size, X_train, y_train, X_test, y_test):
        """
        Run the training process using Keras Tuner.
        """
        self.logger.debug('Run method called.')
        self.train(window_size, X_train, X_test, y_train, y_test)

    def build_model(self, hp):
        """
        Build a Keras model with hyperparameters from Keras Tuner.
        Args:
            hp: HyperParameters object from Keras Tuner.
        Returns:
            Compiled Keras model.
        """
        self.logger.debug('Building model with Keras Tuner hyperparameters.')
        activation_space = {
            'activation1': hp.Choice('activation1', ['relu', 'tanh', 'sigmoid', 'elu']),
            'activation2': hp.Choice('activation2', ['relu', 'tanh', 'sigmoid', 'elu']),
            'activation3': hp.Choice('activation3', ['relu', 'tanh', 'sigmoid', 'elu']),
            'activation4': hp.Choice('activation4', ['relu', 'tanh', 'sigmoid', 'elu']),
            'activation5': hp.Choice('activation5', ['relu', 'tanh', 'sigmoid', 'elu']),
        }
        activations = {key: activation_space[key] for key in activation_space}
        dense_units_space = {
            'dense_units1': hp.Choice('dense_units1', [64, 96, 128, 256]),
            'dense_units2': hp.Choice('dense_units2', [32, 64, 96, 128]),
            'dense_units3': hp.Choice('dense_units3', [8, 16, 32, 64]),
            'dense_units4': hp.Choice('dense_units4', [8, 16, 32, 64]),
        }
        dense_units = {key: dense_units_space[key] for key in dense_units_space}
        dropout_space = {
            'dropout_rate1': hpo.uniform('dropout_rate1', 0.01, 0.3),
            'dropout_rate2': hpo.uniform('dropout_rate2', 0.01, 0.3),
            'dropout_rate3': hpo.uniform('dropout_rate3', 0.01, 0.3),
            'dropout_rate4': hpo.uniform('dropout_rate4', 0.01, 0.3)
        }
        dropout = {key: dropout_space[key] for key in dropout_space}
        bias_space = {
            'bias_term1': hpo.uniform('bias_term1', -1.0, 1.0),
            'bias_term2': hpo.uniform('bias_term2', -1.0, 1.0),
            'bias_term3': hpo.uniform('bias_term3', -1.0, 1.0),
            'bias_term4': hpo.uniform('bias_term4', -1.0, 1.0)
        }
        bias = {key: bias_space[key] for key in bias_space}
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_units['dense_units1'], activation=activations['activation1'], input_shape=(96,)),
            tf.keras.layers.Dense(dense_units['dense_units2'], activation=activations['activation2']),
            tf.keras.layers.Dense(dense_units['dense_units3'], activation=activations['activation3']),
            tf.keras.layers.Dense(dense_units['dense_units4'], activation=activations['activation4']),
            tf.keras.layers.Dense(1, activation=activations['activation5'])
        ])
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])
        if optimizer_choice == 'adam':
            learning_rate = hp.Float('adam_learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            learning_rate = hp.Float('sgd_learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            learning_rate = hp.Float('rmsprop_learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            learning_rate = hp.Float('adadelta_learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        loss_choice = hp.Choice('loss', values=['binary_crossentropy', 'hinge', 'squared_hinge'])
        batch_size = hp.Choice('batch_size', values=[16, 32, 64, 96, 128])
        model.compile(optimizer=optimizer, loss=loss_choice, metrics=['accuracy'])
        self.logger.debug('Model compiled with selected hyperparameters.')
        return model

    def train(self, window_size, X_train, X_test, y_train, y_test, model_save_path: str = None, epochs=50, **kwargs):
        """
        Train a Keras model using Keras Tuner and save the best model.
        Args:
            window_size: Window size for the model.
            X_train, X_test, y_train, y_test: Data splits.
            model_save_path: Path to save the best model.
            epochs: Number of training epochs (from config).
        """
        self.logger.info('Starting Keras Tuner model training.')
        self.logger.info(f'Training parameters: epochs={epochs}')
        # Increase max_trials to test more hyperparameter combinations
        tuner = kt.RandomSearch(self.build_model, objective='val_accuracy', max_trials=10)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])
        self.logger.debug('Hyperparameter search completed.')
        best_model = tuner.get_best_models(num_models=1)[0]
        if model_save_path is None:
            model_save_path = 'model_keras_tuner.h5'
        best_model.save(model_save_path)
        self.logger.info(f'Best model saved to {model_save_path}')

    def validation(self):
        """
        Placeholder for validation logic.
        """
        self.logger.debug('Validation method called (not implemented).') 