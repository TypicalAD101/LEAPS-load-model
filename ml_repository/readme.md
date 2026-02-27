# ML Repository - Time Series Forecasting Pipeline

**Objective**: Common repository for ML model training and validation

This project implements a comprehensive pipeline for time series forecasting using four different models: **Keras**, **Keras Tuner**, **Darts**, and **ARIMA**. The pipeline includes data handling, model training, validation, testing, and implementation stages.

## Python version
Preferred: ver 3.8

## Overview
The `Train` class in `master.py` script controls the entire process. 
It utilizes the following modules for specific tasks:

### Core Modules
1. **`Support/data_handler.py`**: Data loading, preprocessing, visualization, and post-processing.
2. **`Support/config_handler.py`**: Configuration management and parameter handling.
3. **`Support/db_handler.py`**: Database operations for storing validation results.

### Training Modules
4. **`Training/base_trainer.py`**: Base trainer class providing common functionality.
5. **`Training/keras_model_train.py`**: Keras model training implementation.
6. **`Training/keras_tuner_model.py`**: Keras Tuner hyperparameter optimization.
7. **`Training/darts_model.py`**: Darts time series model training.
8. **`Training/arima_model.py`**: ARIMA model training and forecasting.

### Validation Modules
9. **`Validation/base_validator.py`**: Base validator with common validation logic.
10. **`Validation/keras_validator.py`**: Keras model validation.
11. **`Validation/keras_tuner_validator.py`**: Keras Tuner model validation.
12. **`Validation/darts_validator.py`**: Darts model validation.
13. **`Validation/arima_validator.py`**: ARIMA model validation.

### Testing Modules
14. **`Testing/base_tester.py`**: Base tester with common testing functionality.
15. **`Testing/keras_tester.py`**: Keras model testing and evaluation.
16. **`Testing/keras_tuner_tester.py`**: Keras Tuner model testing.
17. **`Testing/darts_tester.py`**: Darts model testing.
18. **`Testing/arima_tester.py`**: ARIMA model testing.

### Implementation Modules
19. **`Implementation/implementation.py`**: Final implementation and prediction generation for all models.

### Utility Modules
20. **`utils/model_enums.py`**: Model type enumerations and configuration keys.
21. **`utils/logger.py`**: Centralized logging functionality.

## Supported Models

The pipeline now supports four different time series forecasting models:

- **Keras (k)**: Deep learning model using LSTM layers
- **Keras Tuner (t)**: Automated hyperparameter optimization with Keras Tuner
- **Darts (d)**: Time series library with N-BEATS and Transformer models
- **ARIMA (a)**: Traditional statistical time series model

## Project Structure

The project is organized into the following folder structure:

```
ml_repository/
├── models/           # Trained model files (.h5, .pkl, .ckpt)
├── reports/          # Performance reports and analysis (.csv)
├── databases/        # Validation and testing databases (.db)
├── logs/             # Pipeline logs and debugging information
├── Training/         # Model training implementations
├── Validation/       # Model validation modules
├── Testing/          # Model testing and evaluation
├── Implementation/   # Final implementation and prediction
├── Support/          # Core utilities (data, config, database handlers)
├── utils/            # Utility modules (enums, logger)
├── UseCase/          # Data files and use case examples
├── other_data/       # Additional data files
└── venv/            # Python virtual environment
```

## Logging
A centralized logger is used throughout the pipeline. Logs are written to both the console and `logs/pipeline.log`.
- Info-level logs for major steps, debug-level for details.
- Set `[Logging] debug = true` in `config.ini` to enable debug-level logs.

## Configuration
The project uses a comprehensive `config.ini` file to manage various parameters and settings:

### Model Configuration
- **Model modes**: `train_mode`, `val_mode`, `impl_mode` (k/t/d/a)
- **Sequence length**: For time series windowing
- **Model file paths**: Paths for saved models
- **Data file path**: Set as `data_path` under `[Paths]`

### Database Configuration
- **Validation databases**: Separate databases for each model type
- **Performance tracking**: Automated result storage

### Training Configuration
- **Batch size**: Training batch size
- **Epochs**: Number of training epochs
- **Validation split**: Percentage for validation
- **Test windows**: Number of test windows for evaluation

### Logging Configuration
- **Debug mode**: Enable/disable detailed logging

The configuration file allows for easy modification of these parameters without changing the code.

## Running the Pipeline

The `Train()` class constructor initiates the pipeline. The `run()` method in `master.py` controls 
which stages of the pipeline are executed. Currently, it calls the implementation stage by default. 
You can uncomment the other method calls in the `run()` method if you want to run 
the complete pipeline stages.

### Model Mode Configuration
Model modes are configured in `config.ini`:
```ini
[Model]
train_mode = k  # 'k' for Keras, 't' for Keras Tuner, 'd' for Darts, 'a' for ARIMA
val_mode = k    # 'k' for Keras, 't' for Keras Tuner, 'd' for Darts, 'a' for ARIMA
impl_mode = k   # 'k' for Keras, 't' for Keras Tuner, 'd' for Darts, 'a' for ARIMA
```

## Pipeline Stages

### 1. Data Handling
- Data loading and preprocessing
- Time series windowing
- Data visualization (optional)

### 2. Model Training
- Model-specific training implementations
- Hyperparameter optimization (Keras Tuner)
- Model saving to specified paths

### 3. Model Validation
- Performance evaluation on validation data
- Metrics calculation (MAE, MSE, MAPE)
- Results storage in databases

### 4. Model Testing
- Comprehensive testing framework
- Performance comparison across models
- Automated test report generation

### 5. Model Implementation
- Rolling window forecasting
- Recursive prediction generation
- Performance report generation

## Code Flow
For the complete training and validation use the following sequence:
```
Train → Validation → Testing → Implementation
```

The sequence can be controlled in the `self.run()` method of the `Train` class in `master.py`.

For individual control, enable training first to generate the model, then use the model for validation, testing, and implementation.

## Testing Framework

The new testing framework provides:
- **Base tester class**: Common testing functionality
- **Model-specific testers**: Specialized testing for each model type
- **Performance metrics**: Comprehensive evaluation metrics
- **Automated reporting**: Test result generation and storage

## Implementation Features

The enhanced implementation module provides:
- **Rolling window forecasting**: Multi-step prediction using sliding windows
- **Recursive forecasting**: Long-term prediction capabilities
- **Performance tracking**: Automated performance measurement
- **Report generation**: Detailed performance reports in multiple formats

## Use Case

This folder contains data and instructions for generating profiles used for prediction tasks. 
The data is structured in a CSV format, with each entry representing a measured value associated 
with a timestamp.

### Data Reference
The `Load_Profile.csv` file can be used as a reference for profile formatting. This is just a reference.
The timestamp format should be similar to the one mentioned in the Load profile. Interval and
data range will be as per requirement of the data.

### Load_Profile.csv Structure
- **Date**: Represents the timestamp for each load entry, generated in 15-minute intervals for an 
entire year (365 days * 24 hours * 4 intervals per hour).
- **crit_load**: Contains the critical load value for each timestamp, 
randomly generated within the range of 240 to 1000.

### Purpose
The synthetic data is intended for use in predicting critical load profiles in various scenarios. 
The data is provided in CSV format, making it easy to import into prediction algorithms.

### Data Structure
- **Date (datetime)**: The timestamp of the load entry, recorded at 15-minute intervals.
- **crit_load (integer)**: A randomly generated load value between 240 and 1000.

### File Location
The `Load_Profile.csv` file should be stored in the `UseCase` folder in this repository, and it can be referenced for training or testing purposes.

### Usage
1. Ensure that any profile file is located in the `UseCase` folder.
2. Load the data into your prediction model using pandas.
3. Utilize the load values and timestamps for analysis and modeling.

## Recent Updates

### ARIMA Model Support
- Added comprehensive ARIMA model training, validation, and testing
- Integrated ARIMA into the main pipeline
- Added ARIMA-specific configuration options

### Enhanced Configuration Management
- Centralized model type management using enums
- Improved configuration parameter organization
- Added database configuration for all model types

### Testing Framework
- New Testing directory with specialized testers for each model
- Automated performance evaluation and reporting
- Comprehensive testing metrics and comparison tools

### Implementation Enhancements
- Improved rolling window forecasting capabilities
- Enhanced recursive prediction generation
- Better performance tracking and reporting

### Logging Improvements
- Enhanced logging throughout the pipeline
- Better traceability for debugging and monitoring
- Configurable debug levels