# CALM: Curiosity-Driven Action-Sensitive Language Models

## Overview

CALM (Curiosity-Driven Action-Sensitive Language Models) is a novel hybrid approach that combines fast LLM scaling with developmental grounding through curiosity-driven action data.

## Core Innovation

**Problem**: 
- Current LLMs scale fast but lack grounding (hallucinations, no true understanding)
- Developmental AI is grounded but slow (can't scale like LLMs)

**Solution**: Train LLMs with curiosity-driven action data to make them action-sensitive/grounded while maintaining fast scaling

## Key Features

- **Fast LLM Scaling**: Uses transformer architecture (not slow developmental timeline)
- **Action Grounding**: Trains on curiosity-driven action data
- **Action-Language Alignment**: Aligns language representations with action sequences
- **Curiosity-Driven Generation**: Uses curiosity as training signal for exploration
- **Complete Mathematical Foundation**: Fully documented mathematical formulations

## Installation

```bash
git clone https://github.com/mbhat24/roformer-llm.git
cd roformer-llm
git checkout calm-research
pip install -r requirements.txt
```

## Quick Start

### 1. Train CALM

```bash
python scripts/train_calm.py
```

This will:
- Collect 100 episodes of action data using curiosity-driven exploration
- Create language-action alignment dataset
- Train CALM model for 5 epochs
- Save checkpoints to `checkpoints/calm/`

### 2. Mathematical Background

See `CALM_MATHEMATICS.md` for complete mathematical formulation including:
- Curiosity computation (KL divergence, information gain)
- Forward model prediction
- Action-language alignment loss
- Complete loss function derivation
- Training dynamics
- Convergence analysis
- Theoretical properties

## Architecture

### Components

1. **Language Encoder**: Transformer-based encoder for text
2. **Action Encoder**: MLP-based encoder for action sequences
3. **Alignment Module**: Aligns language and action representations
4. **Curiosity Module**: Computes curiosity-driven exploration signal
5. **Forward Model**: Predicts next state from action

### Loss Function

```
L_total = L_lm + λ * L_align + γ * L_curiosity + δ * L_forward
```

Where:
- `L_lm`: Language modeling loss
- `L_align`: Action-language alignment loss
- `L_curiosity`: Curiosity loss
- `L_forward`: Forward model loss
- `λ, γ, δ`: Hyperparameters

## Project Structure

```
src/calm/
├── __init__.py           # Module exports
├── maths.py              # Mathematical formulations
├── model.py              # CALM architecture
└── data_collection.py    # Action data collection pipeline

scripts/
└── train_calm.py         # Training script

docs/
├── CALM_RESEARCH.md       # Research plan
└── CALM_MATHEMATICS.md   # Complete mathematical documentation

checkpoints/calm/         # Model checkpoints
data/calm/                 # Action data
```

## Training Results

### Test Run (100 episodes, 5 epochs)

- **Episodes Collected**: 100
- **Total Steps**: 10,058
- **Average Curiosity**: 3.15
- **Final Loss**: 1.12
- **Model Parameters**: 27.6M

### Action Prediction

Model successfully predicts actions from language descriptions:
- Input: "the agent performed 50 actions"
- Predicted action: [-0.84, -0.24, 1.56]

## Novelty

**First of its kind**: No existing work combines curiosity-driven action learning with LLM training in this way.

**Key differences from existing work**:
- Existing: Developmental AI is slow
- CALM: Fast LLM scaling + developmental grounding
- Existing: Action and language learned together
- CALM: Pre-collected action data + LLM training

## Mathematical Foundation

Complete mathematical documentation in `CALM_MATHEMATICS.md`:

1. **Curiosity Computation**: KL divergence, information gain, prediction error
2. **Forward Model**: Next state prediction, training loss
3. **Action-Language Alignment**: Loss formulation, encoder architectures
4. **Complete Loss Function**: Derivation, hyperparameter analysis
5. **Training Dynamics**: Gradient flow, optimizer updates
6. **Computational Complexity**: Time and space analysis
7. **Theoretical Properties**: Convergence theorems, proofs
8. **Evaluation Metrics**: Alignment, curiosity, accuracy
9. **Extensions**: Multi-head curiosity, hierarchical alignment

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| vocab_size | 50000 | Vocabulary size |
| embed_dim | 256 | Embedding dimension |
| num_heads | 8 | Number of attention heads |
| num_layers | 4 | Number of transformer layers |
| action_dim | 3 | Action dimension |
| max_seq_len | 128 | Maximum sequence length |
| lambda_align | 1.0 | Alignment loss weight |
| gamma_curiosity | 0.1 | Curiosity loss weight |
| delta_forward | 0.5 | Forward model loss weight |

## Citation

If you use CALM in your research, please cite:

```bibtex
@article{calm2025,
  title={Curiosity-Driven Action-Sensitive Language Models},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- Inspired by curiosity-driven exploration research
- Built on transformer architecture
- Mathematical foundation from information theory
