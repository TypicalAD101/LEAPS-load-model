# SOTA CNN-LSTM Time Series Forecasting Model

## Overview

This repository contains a state-of-the-art (SOTA) time series forecasting pipeline that implements a CNN-LSTM with Multi-Head Attention architecture. The model is specifically designed for univariate, single-step prediction and is optimized for HPC environments with multi-GPU support.

## 🚀 Key Features

### 1. Advanced Data Processing Pipeline
- **v3 Preprocessing**: Intelligent gap classification and hierarchical filling
- **Data Leakage Prevention**: Proper train/val/test split before scaling
- **Advanced Imputation**: Context-aware missing data handling
- **Data Augmentation**: Gaussian noise injection and amplitude scaling

### 2. SOTA Model Architecture
- **1D CNN Layers**: Local temporal pattern extraction and noise filtering
- **LSTM Layers**: Long-term temporal dependency learning
- **Multi-Head Attention**: Dynamic timestep weighting
- **Residual Connections**: Improved gradient flow
- **Batch Normalization**: Training stability and convergence

### 3. Advanced Training Strategy
- **Custom Training Loop**: Recursive forecasting simulation
- **Advanced LR Scheduling**: Warmup + cosine annealing
- **Early Stopping**: Enhanced with learning rate reduction on plateau
- **Model Checkpointing**: Automatic best model saving
- **Data Augmentation**: Real-time augmentation during training

### 4. HPC Optimizations
- **Multi-GPU Support**: MirroredStrategy for data-parallel training
- **Mixed Precision**: FP16 training for memory efficiency
- **XLA Compilation**: Accelerated Linear Algebra optimization
- **Advanced TF Optimizations**: Layout, constant folding, and more
- **Memory Management**: Optimized for 80GB A100 GPUs
- **Smart NCCL Configuration**: Respects cluster defaults for optimal performance

### 5. Hyperparameter Optimization
- **Keras Tuner Integration**: Automatic hyperparameter search
- **Hyperband Algorithm**: Efficient trial scheduling
- **Comprehensive Search Space**: CNN filters, LSTM units, attention heads

## 🏗️ Architecture Details

```
Input: (batch_size, 96, 1)
    ↓
1D CNN Layers (64 filters → 128 filters)
    ↓
MaxPooling1D (pool_size=2)
    ↓
LSTM Layers (256 → 320 → 128 units)
    ↓
Multi-Head Attention (8 heads)
    ↓
Residual Connection + Layer Normalization
    ↓
Global Average Pooling
    ↓
Dense Layers (256 → 128 units)
    ↓
Output: (batch_size, 1)
```

## 📊 Data Processing Pipeline

### v3 Preprocessing Features
1. **Gap Classification**: Short (<7h) vs Long (≥7h) gaps
2. **Short Gap Filling**: Hierarchical approach (D-1 → D-7 → ToD median → LOCF)
3. **Long Gap Filling**: Time-of-Day median for stability
4. **Zero Handling**: Converts zeros to NaN for proper gap detection

### Data Augmentation
- **Gaussian Noise**: Small noise injection (std=0.01)
- **Amplitude Scaling**: Random scaling (0.95x to 1.05x)
- **Value Clipping**: Prevents extreme values

## 🚀 Usage

### SLURM HPC Environment (Recommended)
The repository includes SLURM scripts optimized for HPC clusters:

```bash
# 1. Create environment
sbatch 01_make_env.slurm

# 2. Test GPU setup
sbatch 02_probe_gpu.slurm

# 3. Run hyperparameter optimization
sbatch 03_train_keras_tuner_gpu.slurm

# 4. Train final model with best hyperparameters
sbatch 04_train_best_hyper_params.slurm
```

### Direct Python Execution
```bash
python train.py \
    --data_dir /path/to/csv/files \
    --output_dir /path/to/output \
    --logs_dir /path/to/logs \
    --epochs 200 \
    --batch_size 256
```

### Hyperparameter Optimization
```bash
python train.py \
    --data_dir /path/to/csv/files \
    --output_dir /path/to/output \
    --logs_dir /path/to/logs \
    --use_tuner \
    --max_trials 30 \
    --epochs_per_trial 50
```

### Advanced Options
```bash
python train.py \
    --data_dir /path/to/csv/files \
    --output_dir /path/to/output \
    --logs_dir /path/to/logs \
    --epochs 200 \
    --batch_size 1024 \
    --patience 25 \
    --learning_rate 0.0005 \
    --enable_augmentation \
    --mixed_precision \
    --xla_compilation
```

## 📁 Output Structure

```
output_dir/
├── best_tuned_model.h5          # Best model from hyperparameter search
├── final_model.h5               # Final trained model
├── best_hyperparameters.json    # Best hyperparameters found
├── performance_metrics.json      # Model performance metrics
├── prediction_analysis.json     # Detailed prediction analysis
├── data_overview.png            # Data visualization
├── training_history.png         # Training curves
├── advanced_predictions_analysis.png  # Comprehensive prediction analysis
└── model_checkpoints/           # Training checkpoints
    └── best_model_weights.h5   # Best weights during training
```

## 🔧 Requirements

### Core Dependencies
- TensorFlow 2.12+
- NumPy 1.21+
- Pandas 1.3+
- Scikit-learn 1.0+
- Matplotlib 3.5+
- Keras Tuner 1.3+

### HPC Cluster Configuration
- **Modern HPC Cluster**: Optimized for systems like ASU Sol supercomputer
- **Smart NCCL Handling**: Uses cluster defaults for optimal performance
- **Configurable Troubleshooting**: NCCL settings can be enabled if needed
- **Multi-GPU Ready**: Tested with 2x A100 80GB configurations

### SLURM Scripts
The repository includes optimized SLURM scripts for HPC environments:
- **`01_make_env.slurm`**: Creates conda environment with all dependencies
- **`02_probe_gpu.slurm`**: Tests GPU setup and CUDA configuration
- **`03_train_keras_tuner_gpu.slurm`**: Runs hyperparameter optimization (15 trials, 25 epochs each)
- **`04_train_best_hyper_params.slurm`**: Trains final model with best hyperparameters (200 epochs)

### GPU Requirements
- NVIDIA GPU with CUDA support
- Minimum 16GB VRAM (recommended: 80GB A100)
- CUDA 11.8+ and cuDNN 8.6+

### HPC Environment
- Multi-GPU node support
- NCCL for GPU communication
- Sufficient system memory (128GB+ recommended)

## 📈 Performance Features

### Training Optimizations
- **Scheduled Sampling**: Gradual transition from ground truth to predictions
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Warmup**: Stable early training
- **Cosine Annealing**: Smooth learning rate decay

### Evaluation Metrics
- **MSE/RMSE**: Standard regression metrics
- **MAE**: Mean absolute error
- **MAPE**: Mean absolute percentage error
- **Correlation**: Prediction vs actual correlation
- **Residual Analysis**: Comprehensive error analysis

## 🎯 Use Cases

### Primary Applications
- **Load Forecasting**: Residential, commercial, industrial
- **Energy Demand Prediction**: Grid management and planning
- **Time Series Forecasting**: Any univariate sequential data
- **HPC Research**: Multi-GPU training research

### Deployment Integration
- **forecast_module**: Compatible with existing recursive forecasting
- **Real-time Prediction**: Single-step prediction capability
- **Batch Processing**: Efficient batch inference
- **Model Serving**: TensorFlow SavedModel format

## 🔬 Research Features

### Advanced Techniques
- **Transfer Learning**: Pre-training on combined datasets
- **Domain Adaptation**: Fine-tuning for specific data types
- **Ensemble Methods**: Multiple model combination
- **Cross-Validation**: Robust performance estimation

### Experimental Features
- **Attention Visualization**: Understanding model focus
- **Feature Importance**: CNN filter analysis
- **Gradient Analysis**: Training dynamics
- **Memory Profiling**: GPU utilization optimization

## 📚 References

### Architecture Papers
- "Attention Is All You Need" - Vaswani et al.
- "Deep Learning for Time Series Forecasting" - Various authors
- "CNN-LSTM Hybrid Models" - Recent research papers

### Implementation Guides
- TensorFlow Multi-GPU Training
- Keras Tuner Best Practices
- HPC Optimization Techniques

## 🤝 Contributing

We welcome contributions to improve this SOTA implementation:

1. **Architecture Improvements**: Novel attention mechanisms, CNN designs
2. **Training Optimizations**: Better scheduling, regularization techniques
3. **Data Processing**: Enhanced augmentation, preprocessing methods
4. **HPC Features**: Better multi-GPU strategies, memory optimization
5. **Documentation**: Usage examples, performance benchmarks

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent framework
- Keras Tuner developers for hyperparameter optimization
- HPC community for multi-GPU training techniques
- Research community for SOTA time series methods

---

**Note**: This implementation represents the current state-of-the-art in time series forecasting. For production use, ensure proper validation and testing on your specific datasets. 