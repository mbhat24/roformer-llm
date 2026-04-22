# RoFormer LLM

A decoder-only transformer language model enhanced with Rotary Position Embeddings (RoPE) from the RoFormer paper.

## Overview

This project implements a GPT-style autoregressive language model that uses RoFormer's rotary position embeddings instead of traditional absolute position embeddings. RoPE encodes position information by rotating the query and key vectors, which naturally incorporates relative position dependency in the attention mechanism.

## Architecture

- **Rotary Position Embeddings (RoPE)**: Position information is encoded by rotating query and key vectors based on their absolute positions
- **Multi-Head Attention**: Standard transformer attention enhanced with RoPE
- **Transformer Blocks**: Pre-norm architecture with attention and feed-forward layers
- **Language Modeling Head**: Projects to vocabulary for next-token prediction

## Key Features

- **Relative Position Encoding**: RoPE naturally encodes relative positions through rotation
- **Sequence Length Flexibility**: No maximum sequence length constraint
- **Long-Term Decay**: Attention weights decay with relative distance
- **Linear Attention Compatible**: Can work with linear attention mechanisms

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

This will:
1. Build a vocabulary from sample texts
2. Create and train the model
3. Save the checkpoint to `roformer_llm.pt`
4. Test text generation

### Interactive Demo

```bash
python demo.py
```

### Batch Generation

```bash
python demo.py --batch
```

## Model Architecture

The model consists of:

1. **Token Embeddings**: Convert token IDs to dense vectors
2. **Transformer Blocks**: Multiple layers of:
   - Multi-Head Attention with RoPE
   - Feed-Forward Network
   - Layer Normalization
   - Residual Connections
3. **Language Modeling Head**: Project to vocabulary logits

## RoPE Mathematics

The rotary position embedding rotates the query and key vectors:

```
f_q(x_m, m) = R^d_Θ,m W_q x_m
f_k(x_n, n) = R^d_Θ,n W_k x_n
```

Where R^d_Θ,m is the rotation matrix:

```
R^d_Θ,m = diag(Rotation(mθ_1), Rotation(mθ_2), ..., Rotation(mθ_d/2))
```

With θ_i = 10000^(-2(i-1)/d)

The attention computation becomes:

```
q_m^T k_n = x_m^T W_q^T R^d_Θ,n-m W_k x_n
```

This depends only on relative position (n-m).

## Configuration

Default configuration:
- Embedding dimension: 256
- Attention heads: 8
- Transformer layers: 6
- Feed-forward dimension: 1024
- Maximum sequence length: 512
- Dropout: 0.1

## Example

```python
from model import RoFormerLLM
from tokenizer import SimpleTokenizer

# Create tokenizer
tokenizer = SimpleTokenizer(texts)

# Create model
model = RoFormerLLM(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_position_embeddings=512
)

# Generate text
prompt = "The attention mechanism"
prompt_ids = tokenizer.encode(prompt).unsqueeze(0)
generated = model.generate(prompt_ids, max_new_tokens=50)
text = tokenizer.decode(generated[0])
```

## References

- RoFormer Paper: https://arxiv.org/abs/2104.09864
- Original Transformer: https://arxiv.org/abs/1706.03762

## License

MIT License
