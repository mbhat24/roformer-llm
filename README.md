# RoFormer LLM

A professional decoder-only transformer language model enhanced with Rotary Position Embeddings (RoPE) from the RoFormer paper.

## Overview

This project implements a production-ready GPT-style autoregressive language model that uses RoFormer's rotary position embeddings. RoPE encodes position information by rotating query and key vectors, naturally incorporating relative position dependency in the attention mechanism.

## Project Structure

```
roformer-llm/
├── src/
│   └── roformer/
│       ├── model/          # Model components (RoPE, attention, LLM)
│       ├── data/           # Data loading, tokenization, preprocessing
│       ├── training/       # Training utilities (checkpointing)
│       ├── utils/          # Configuration, logging, hardware detection
│       └── evaluation/     # Benchmarks and evaluation metrics
├── configs/                # YAML configuration files
├── scripts/                # Training and evaluation scripts
├── data/                   # Data storage (raw and processed)
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Key Features

- **Rotary Position Embeddings (RoPE)**: Mathematically correct implementation from RoFormer paper
- **Hardware Auto-Detection**: Automatically optimizes for CUDA, MPS (Apple Silicon), or CPU
- **Professional Training Loop**: Learning rate scheduling, checkpointing, logging, validation
- **Custom Data Support**: Load data from files, URLs, directories, or direct input
- **Configuration Management**: YAML-based configuration system
- **Standard Benchmarks**: Perplexity, generation speed, memory usage, coherence
- **Production Ready**: Proper logging, checkpointing, and error handling

## Installation

```bash
# Clone repository
git clone https://github.com/mbhat24/roformer-llm.git
cd roformer-llm

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### 1. Prepare Data

```bash
# Download sample data (Shakespeare)
python scripts/download_data.py

# Or use custom data
python -c "from roformer.data import CustomDataLoader; chunks = CustomDataLoader.from_file('your_data.txt')"
```

### 2. Configure Training

Edit `configs/default.yaml` to customize:
- Model architecture (layers, dimensions, heads)
- Training parameters (batch size, learning rate, epochs)
- Data settings (max length, splits)
- Hardware configuration

### 3. Train Model

```bash
# Deep training with professional loop
python scripts/train_deep.py

# Or use fast training for quick experiments
python scripts/train_fast.py
```

### 4. Evaluate

```bash
# Run benchmarks
python scripts/benchmarks.py
```

### 5. Generate Text

```bash
# Interactive demo
python scripts/demo.py

# Batch generation
python scripts/demo.py --batch
```

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
model:
  embed_dim: 512
  num_heads: 8
  num_layers: 8
  max_position_embeddings: 512

training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 3e-5

data:
  data_path: "data/raw/shakespeare.txt"
  max_length: 256
```

## Hardware Optimization

The system automatically detects and optimizes for available hardware:

- **CUDA (NVIDIA)**: Full GPU acceleration
- **MPS (Apple Silicon)**: Metal Performance Shaders
- **CPU**: Fallback with optimized settings

Run hardware detection:
```bash
python -c "from roformer.utils import detect_hardware; detect_hardware()"
```

## API Usage

```python
from roformer import RoFormerLLM, RoFormerLLMConfig, WordTokenizer
from roformer.utils import Config

# Load configuration
config = Config.from_yaml("configs/default.yaml")

# Create model
model = RoFormerLLM(
    vocab_size=config.model.vocab_size,
    embed_dim=config.model.embed_dim,
    num_heads=config.model.num_heads,
    num_layers=config.model.num_layers
)

# Generate text
tokenizer = WordTokenizer(texts)
prompt_ids = tokenizer.encode("To be, or not to be")
generated = model.generate(prompt_ids, max_new_tokens=50)
```

## Benchmarks

### Latest Training Results (27.8M parameter model, 50 epochs)

- **Perplexity**: 94.41 (46% improvement over previous)
- **Generation Speed**: 30.0 tokens/sec (MPS)
- **Memory Overhead**: -7.6 MB (efficient)
- **Coherence Score**: 0.814
- **Validation Loss**: 4.6676

See `TRAINING_RESULTS.md` for detailed analysis and training progress.

### Previous Results (12.5M parameter model, 5 epochs)

- **Perplexity**: 174.34
- **Generation Speed**: 40.8 tokens/sec (MPS)
- **Memory Overhead**: 0.2 MB
- **Coherence Score**: 0.886

See `BENCHMARK_REPORT.md` for detailed analysis.

## RoPE Mathematics

The rotary position embedding rotates query and key vectors:

```
f_q(x_m, m) = R^d_Θ,m W_q x_m
f_k(x_n, n) = R^d_Θ,n W_k x_n
```

Where R^d_Θ,m is the rotation matrix with θ_i = 10000^(-2(i-1)/d).

The attention computation depends only on relative position:
```
q_m^T k_n = x_m^T W_q^T R^d_Θ,n-m W_k x_n
```

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ scripts/

# Type check
mypy src/
```

## References

- RoFormer Paper: https://arxiv.org/abs/2104.09864
- Original Transformer: https://arxiv.org/abs/1706.03762

## License

MIT License
