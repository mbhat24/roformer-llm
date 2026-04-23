# RoFormer LLM & DevelAI

A professional decoder-only transformer language model enhanced with Rotary Position Embeddings (RoPE) from the RoFormer paper, plus DevelAI - a novel developmental intelligence architecture.

## Overview

This project has two main components:

### 1. RoFormer LLM
A production-ready GPT-style autoregressive language model that uses RoFormer's rotary position embeddings. RoPE encodes position information by rotating query and key vectors, naturally incorporating relative position dependency in the attention mechanism.

### 2. DevelAI (Developmental Intelligence Architecture)
A radical paradigm shift: building intelligence BEFORE language, mimicking biological development. Unlike current AI that starts with language and tries to build "understanding," DevelAI builds foundational intelligence (motor, sensory, spatial) first, then adds language as a communication interface.

**Key Insight**: Language is a tool for communication, not the foundation of intelligence. A child learns movement and spatial awareness in the womb, then learns words later. DevelAI follows this biological trajectory.

## DevelAI Developmental Stages

DevelAI follows a biological developmental trajectory:

### Stage 1: Motor Learning (Current - In Progress)
- **Goal**: Learn to control a robotic arm in a physics simulation
- **Components**: MotorCortex, Cerebellum, BasalGanglia
- **Task**: Object manipulation (reach, grasp, move objects)
- **Status**: Training overnight (10,000 episodes)
- **No language yet** - just pure motor intelligence

### Stage 2: Cognitive Development (Planned)
- **Goal**: Build planning, memory, causal reasoning
- **Components**: Memory systems, attention mechanisms, decision making
- **Tasks**: Complex problem solving, multi-step planning
- **Still no language** - building cognitive intelligence

### Stage 3: Language Interface (Planned)
- **Goal**: Add language as communication layer
- **How words fit**: Language maps internal representations to words
- **How sentences fit**: Sentences express complex internal states and intents
- **Key difference**: Language is OUTPUT/INPUT, not the intelligence itself
- **Example**: The system "knows" how to grasp an object (Stage 1), then learns to say "I'm grasping the object" (Stage 3)

**Why this works**: The system already has intelligence (can do tasks). Language is just a way to communicate what it's doing/thinking. This is how humans work - we can do things before we can describe them in words.

## Project Structure

```
roformer-llm/
├── src/
│   ├── roformer/           # RoFormer LLM components
│   │   ├── model/          # Model components (RoPE, attention, LLM)
│   │   ├── data/           # Data loading, tokenization, preprocessing
│   │   ├── training/       # Training utilities (checkpointing)
│   │   ├── utils/          # Configuration, logging, hardware detection
│   │   └── evaluation/     # Benchmarks and evaluation metrics
│   └── develai/            # DevelAI developmental intelligence
│       ├── simulation.py   # Physics simulation environment
│       ├── motor.py        # Motor control networks
│       ├── sensory.py      # Sensory processing (planned)
│       ├── cognitive.py    # Cognitive development (planned)
│       └── language.py     # Language interface (planned)
├── configs/                # YAML configuration files
├── scripts/                # Training and evaluation scripts
│   ├── train_deep.py       # RoFormer training
│   └── train_motor.py      # DevelAI motor learning
├── data/                   # Data storage (raw and processed)
├── checkpoints/            # Model checkpoints
│   └── develai/            # DevelAI motor learning checkpoints
├── logs/                   # Training logs
├── results/                # Training results
│   └── develai/            # DevelAI training data
├── tests/                  # Unit tests
└── docs/                   # Documentation
    ├── DEVELAI_RESEARCH.md # DevelAI research plan
    ├── MLPE_RESEARCH.md    # Meta-Learned PE research
    └── COMMLM_RESEARCH.md  # Communication-Centric LM research
```

## Key Features

### RoFormer LLM
- **Rotary Position Embeddings (RoPE)**: Mathematically correct implementation from RoFormer paper
- **Hardware Auto-Detection**: Automatically optimizes for CUDA, MPS (Apple Silicon), or CPU
- **Professional Training Loop**: Learning rate scheduling, checkpointing, logging, validation
- **Custom Data Support**: Load data from files, URLs, directories, or direct input
- **Configuration Management**: YAML-based configuration system
- **Standard Benchmarks**: Perplexity, generation speed, memory usage, coherence
- **Production Ready**: Proper logging, checkpointing, and error handling

### DevelAI
- **Biologically-Inspired Development**: Motor → Cognitive → Language
- **Physics Simulation**: 2D environment for embodied learning
- **Neural Motor Control**: MotorCortex, Cerebellum, BasalGanglia networks
- **Intelligence First**: Builds capabilities before adding language
- **Language as Interface**: Words/sentences map to existing intelligence

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

### 6. Train DevelAI (Developmental Intelligence)

```bash
# Stage 1: Motor learning
python scripts/train_motor.py

# This trains the motor system to control a robotic arm
# Checkpoints saved to checkpoints/develai/
# Results saved to results/develai/training_data.txt
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
