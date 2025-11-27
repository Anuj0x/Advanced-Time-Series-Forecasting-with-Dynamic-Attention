# ChronoMind: Advanced Time Series Forecasting with Dynamic Attention

A cutting-edge, production-ready implementation of Dynamic Attention Networks for multivariate time series forecasting. Built with modern deep learning frameworks, featuring unified architecture, hierarchical attention mechanisms, and optimized performance for real-world applications.

**Creator**: [Anuj0x](https://github.com/Anuj0x) - Expert in Programming & Scripting Languages, Deep Learning & State-of-the-Art AI Models, Generative Models & Autoencoders, Advanced Attention Mechanisms & Model Optimization, Multimodal Fusion & Cross-Attention Architectures, Reinforcement Learning & Neural Architecture Search, AI Hardware Acceleration & MLOps, Computer Vision & Image Processing, Data Management & Vector Databases, Agentic LLMs & Prompt Engineering, Forecasting & Time Series Models, Optimization & Algorithmic Techniques, Blockchain & Decentralized Applications, DevOps, Cloud & Cybersecurity, Quantum AI & Circuit Design, Web Development Frameworks.

## ✨ Key Features

- **Unified Architecture**: Single configurable model supporting hierarchical and max-pooling attention variants
- **Modern Deep Learning Stack**: PyTorch 2.x + PyTorch Lightning + Hydra configuration
- **Advanced Attention Mechanisms**: Dynamic hierarchical attention with learned and mean-based fusion
- **Production Ready**: Type-safe, modular, and optimized for modern hardware
- **Experiment Management**: Built-in TensorBoard logging and Weights & Biases integration
- **Scalable Training**: Automatic mixed precision, distributed training support
- **Comprehensive Evaluation**: MSE, weighted MSE, and custom forecasting metrics

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd chrono-mind

# Install dependencies
pip install -e .
pip install -e ".[dev]"  # For development tools
```

## 🏃 Quick Start

### Training

```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py model.hidden_size=128 training.num_epochs=50

# Train with max-pooling attention variant
python train.py model.max_pool_attention=true
```

### Testing

```bash
# Run tests
python test.py
```

## ⚙️ Configuration

The project uses Hydra for flexible configuration management. Key options:

### Model Architecture

```yaml
model:
  rnn_type: GRU  # GRU, LSTM, RNN
  hidden_size: 96
  num_layers: 1
  dropout: 0.0
  hierarchical_attention: true
  max_pool_attention: false  # Enable for alternative attention mechanism
  attention_method: mean  # mean or learn
  sequence_length: 90
```

### Data Processing

```yaml
data:
  dataset_name: ghl_small  # ghl_small, tep, electricity
  train_file: train_200000_seed_11_vars_23.csv
  sliding_window: false
  normalize: true
```

### Training Pipeline

```yaml
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  val_split: 0.2
  output_dir: outputs
  log_dir: logs
```

## 📁 Project Architecture

```
├── src/dynamic_attention_networks/
│   ├── config/          # Configuration dataclasses
│   ├── models/          # Neural network architectures
│   ├── data/            # Data loading and preprocessing
│   ├── utils/           # Metrics and utilities
│   └── __init__.py
├── conf/                # Hydra configuration files
├── train.py             # Main training script
├── test.py              # Testing suite
└── pyproject.toml       # Modern dependency management
```

## 💻 API Usage

```python
from dynamic_attention_networks import DynamicAttentionNetwork, TimeSeriesDataModule
from dynamic_attention_networks.config import Config

# Create configuration
config = Config()
config.model.hidden_size = 128
config.model.max_pool_attention = True

# Setup data pipeline
data_module = TimeSeriesDataModule(config)
data_module.setup()

# Initialize model
model = DynamicAttentionNetwork(config.model)

# Training with PyTorch Lightning
from dynamic_attention_networks.models import DynamicAttentionModule
lightning_module = DynamicAttentionModule(config)
```

## 🔬 Technical Innovations

1. **Unified Model Architecture**: Single `DynamicAttentionNetwork` class supporting multiple attention variants
2. **Hierarchical Attention**: Multi-level attention mechanism with dynamic context fusion
3. **Modern Training Pipeline**: PyTorch Lightning with automatic optimization and logging
4. **Type-Safe Implementation**: Comprehensive type hints and modern Python patterns
5. **Modular Design**: Clean separation of concerns for maintainability
6. **Performance Optimization**: Automatic mixed precision and hardware acceleration

## 📊 Supported Datasets

- **GHL Dataset**: Greenhouse climate control and monitoring data
- **TEP Dataset**: Tennessee Eastman Process industrial simulation data
- **Electricity Dataset**: Power consumption and load forecasting data

## 📈 Evaluation Metrics

```python
from dynamic_attention_networks.utils import calculate_mse_wmse

# Calculate comprehensive forecasting metrics
results = calculate_mse_wmse(results_dir, sequence_length=90)
print(f"MSE: {results['mse']:.5f}, WMSE: {results['wmse']:.5f}")
```

## 🤝 Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Run the test suite: `python test.py`
3. Format code: `black src/ && isort src/`
4. Type check: `mypy src/`

## 📄 License

[License information]

## 📚 Citation

If you use this code in your research or applications, please cite the original Dynamic Attention Networks work and this modernized implementation.

---

## 💡 Suggested Alternative Project Name & Description

**Project Name**: **TempusNet** - Temporal Intelligence Networks

**Description**: TempusNet is a state-of-the-art deep learning framework for temporal pattern recognition and forecasting. Leveraging advanced attention mechanisms and hierarchical neural architectures, TempusNet provides unified, scalable solutions for multivariate time series analysis across industrial, environmental, and financial domains. Built with modern MLOps practices, it offers seamless experimentation, deployment, and monitoring capabilities for production-grade temporal intelligence applications.

**Why this name?**
- **Tempus**: Latin for "time" - emphasizes the temporal nature
- **Net**: Indicates the neural network architecture
- More memorable and professional than the original name
- Suggests intelligence and networking of temporal data
- Easier to remember and search for

**Tagline**: "Intelligent Temporal Pattern Recognition & Forecasting Framework"
