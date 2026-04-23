# CALM: Complete Mathematical Formulation

## 1. Curiosity Computation

### 1.1 Information Gain (KL Divergence)

**Definition**: Curiosity is computed as the information gain from observing the next state.

**Mathematical Formulation**:
```
C(s_t, a_t, s_{t+1}) = D_KL[q(z_t|o_t, h_{t-1}) || p(z_t|h_{t-1})]
```

Where:
- `s_t`: Current state at time t
- `a_t`: Action taken at time t
- `s_{t+1}`: Next state after taking action
- `z_t`: Latent variable representing the state
- `q(z_t|o_t, h_{t-1})`: Posterior distribution over latent variables given observation and history
- `p(z_t|h_{t-1})`: Prior distribution over latent variables given history
- `D_KL`: Kullback-Leibler divergence

**KL Divergence Computation**:
```
D_KL[q || p] = 0.5 * (log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²) / σ_p² - 1)
```

Where:
- `μ_q, σ_q²`: Mean and variance of posterior distribution
- `μ_p, σ_p²`: Mean and variance of prior distribution

**Implementation**:
```python
def compute_curiosity(state, action, next_state):
    # Compute prior parameters: p(z|h_{t-1})
    prior_mean, prior_log_std = prior_net(state)
    
    # Compute posterior parameters: q(z|o_t, h_{t-1})
    posterior_mean, posterior_log_std = posterior_net(concat([state, action]))
    
    # Compute KL divergence
    prior_std = exp(prior_log_std)
    posterior_std = exp(posterior_log_std)
    
    kl_div = 0.5 * (
        prior_log_std - posterior_log_std +
        (posterior_std² + (posterior_mean - prior_mean)²) / prior_std² - 1
    )
    
    curiosity = sum(kl_div)  # Sum over latent dimensions
    return curiosity
```

### 1.2 Forward Model Prediction Error

**Alternative Curiosity**: Prediction error from forward model.

**Mathematical Formulation**:
```
C_pred(s_t, a_t, s_{t+1}) = ||s_{t+1} - f(s_t, a_t)||²
```

Where:
- `f(s_t, a_t)`: Forward model predicting next state
- `||·||²`: L2 norm

### 1.3 Information Gain (Entropy-Based)

**Mathematical Formulation**:
```
IG(s_t, a_t) = H(s_t) - H(s_t|a_t)
```

Where:
- `H(s_t)`: Entropy of current state
- `H(s_t|a_t)`: Conditional entropy of state given action

**Entropy Estimation**:
```
H(s) = -∫ p(s) log p(s) ds
```

---

## 2. Forward Model

### 2.1 Next State Prediction

**Mathematical Formulation**:
```
s_{t+1} = f(s_t, a_t) + ε
```

Where:
- `f`: Neural network forward model
- `ε ~ N(0, σ²)`: Gaussian noise

**Forward Model Architecture**:
```
f(s, a) = W_3 * ReLU(W_2 * ReLU(W_1 * [s; a] + b_1) + b_2) + b_3
```

Where:
- `[s; a]`: Concatenation of state and action
- `W_i, b_i`: Weight matrices and biases
- `ReLU`: Rectified linear unit activation

**Training Loss**:
```
L_forward = E[(s_{t+1} - f(s_t, a_t))²]
```

---

## 3. Action-Language Alignment

### 3.1 Alignment Loss

**Mathematical Formulation**:
```
L_align = ||f_lang(text) - f_action(action_sequence)||²
```

Where:
- `f_lang`: Language encoder mapping text to embedding space
- `f_action`: Action encoder mapping action sequence to embedding space
- `||·||²`: L2 norm

**Projection to Common Space**:
```
z_lang = W_lang * f_lang(text)
z_action = f_action(action_sequence)

L_align = ||z_lang - z_action||²
```

### 3.2 Language Encoder

**Transformer-based Encoder**:
```
E(text) = Transformer(Embedding(text) + PositionalEmbedding(text))
```

Where:
- `Embedding`: Token embedding matrix
- `PositionalEmbedding`: Sinusoidal position encoding
- `Transformer`: Multi-layer transformer encoder

**Position Encoding**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where:
- `pos`: Position index
- `i`: Dimension index
- `d`: Model dimension

### 3.3 Action Encoder

**MLP-based Encoder**:
```
E(action_seq) = Pool(MLP(action_seq))
```

Where:
- `MLP`: Multi-layer perceptron
- `Pool`: Pooling operation (mean, max, or attention)

---

## 4. Complete CALM Loss Function

### 4.1 Total Loss

**Mathematical Formulation**:
```
L_total = L_lm + λ * L_align + γ * L_curiosity + δ * L_forward
```

Where:
- `L_lm`: Language modeling loss
- `L_align`: Action-language alignment loss
- `L_curiosity`: Curiosity loss
- `L_forward`: Forward model loss
- `λ, γ, δ`: Hyperparameters weighting each component

### 4.2 Language Modeling Loss

**Cross-Entropy Loss**:
```
L_lm = -∑_{t=1}^T log P(w_t | w_{<t})
```

Where:
- `w_t`: Token at position t
- `P(w_t | w_{<t})`: Probability of token given context

**Implementation**:
```python
L_lm = CrossEntropyLoss(logits, labels, ignore_index=-100)
```

### 4.3 Curiosity Loss

**Two Variants**:

**Variant 1: Information Gain**:
```
L_curiosity = -C(s_t, a_t, s_{t+1})
```

**Variant 2: Encourage High Curiosity**:
```
L_curiosity = -log(1 + C(s_t, a_t, s_{t+1}))
```

### 4.4 Forward Model Loss

**MSE Loss**:
```
L_forward = E[(s_{t+1} - f(s_t, a_t))²]
```

---

## 5. Motor Entropy Loss

### 5.1 Entropy Computation

**Mathematical Formulation**:
```
H(π(a|h)) = -∑_a π(a|h) log π(a|h)
```

Where:
- `π(a|h)`: Policy probability distribution
- `h`: History/context

### 5.2 Motor Entropy Loss

**Mathematical Formulation**:
```
L_entropy = -α * H(π(a|h))
```

Where:
- `α`: Entropy coefficient

**Purpose**: Encourage diverse actions during exploration

---

## 6. Action Prediction

### 6.1 Action Prediction from Text

**Mathematical Formulation**:
```
a = g(f_lang(text))
```

Where:
- `f_lang`: Language encoder
- `g`: Action prediction head (MLP)

**Implementation**:
```python
action = action_head(language_encoder(text))
```

---

## 7. Generation with Curiosity

### 7.1 Curiosity-Driven Generation

**Mathematical Formulation**:
```
w_{t+1} = argmax_{w} [P(w|w_{<t}) + β * C(w_{<t}, w)]
```

Where:
- `P(w|w_{<t})`: Language model probability
- `C(w_{<t}, w)`: Curiosity score for generating w
- `β`: Curiosity weight

**Implementation**:
```python
def generate_with_curiosity(prompt, curiosity_threshold):
    for _ in range(max_tokens):
        # Get LM logits
        logits = lm_model(prompt)
        
        # Compute curiosity for each candidate
        candidates = sample(logits, k=10)
        curiosity_scores = [compute_curiosity(c) for c in candidates]
        
        # Select based on curiosity threshold
        if max(curiosity_scores) > curiosity_threshold:
            selected = candidates[argmax(curiosity_scores)]
        else:
            selected = sample(logits, k=1)
        
        prompt = append(prompt, selected)
```

---

## 8. Training Dynamics

### 8.1 Gradient Flow

**Total Loss Gradient**:
```
∇L_total = ∇L_lm + λ * ∇L_align + γ * ∇L_curiosity + δ * ∇L_forward
```

### 8.2 Update Rules

**AdamW Optimizer**:
```
m_t = β_1 * m_{t-1} + (1 - β_1) * ∇L
v_t = β_2 * v_{t-1} + (1 - β_2) * (∇L)²

m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε) - η * λ * θ_t
```

Where:
- `m_t, v_t`: First and second moment estimates
- `β_1, β_2`: Exponential decay rates
- `η`: Learning rate
- `λ`: Weight decay
- `ε`: Small constant for numerical stability

---

## 9. Computational Complexity

### 9.1 Time Complexity

**Curiosity Computation**: O(d) where d is latent dimension
**Forward Model**: O(d_s * d_a * h) where d_s is state dim, d_a is action dim, h is hidden dim
**Alignment Loss**: O(d_lang * d_action) where d_lang is language dim, d_action is action dim
**Total per Batch**: O(B * (d + d_s * d_a * h + d_lang * d_action)) where B is batch size

### 9.2 Space Complexity

**Model Parameters**: O(d_lang² + d_action² + d_latent²)
**Batch Memory**: O(B * (d_lang + d_action + d_latent))

---

## 10. Theoretical Properties

### 10.1 Curiosity as Exploration Signal

**Theorem**: Curiosity computed as KL divergence is equivalent to expected information gain.

**Proof**:
```
I(s_t; s_{t+1} | a_t) = H(s_{t+1} | a_t) - H(s_{t+1} | s_t, a_t)
                   = D_KL[p(s_{t+1}|s_t, a_t) || p(s_{t+1}|a_t)]
```

### 10.2 Alignment Convergence

**Theorem**: If language and action encoders are sufficiently expressive, alignment loss converges to zero.

**Proof Sketch**: By universal approximation theorem, neural networks can approximate any continuous function. With sufficient capacity, encoders can learn isomorphic mappings to a common latent space.

---

## 11. Hyperparameter Sensitivity

### 11.1 Loss Weights

**λ (Alignment Weight)**:
- Too low: No action-language alignment
- Too high: Language generation suffers

**γ (Curiosity Weight)**:
- Too low: No curiosity-driven exploration
- Too high: Unstable, random generations

**δ (Forward Weight)**:
- Too low: No forward model learning
- Too high: Overfits to action prediction

### 11.2 Learning Rate

**Recommended Range**: 1e-5 to 1e-3
- Lower: More stable, slower convergence
- Higher: Faster convergence, potential instability

---

## 12. Evaluation Metrics

### 12.1 Alignment Score

**Mathematical Formulation**:
```
Alignment = 1 - (1/N) ∑_{i=1}^N ||f_lang(text_i) - f_action(action_i)||² / ||f_lang(text_i)||²
```

### 12.2 Curiosity Score

**Mathematical Formulation**:
```
Avg_Curiosity = (1/N) ∑_{i=1}^N C(s_i, a_i, s_{i+1})
```

### 12.3 Action Prediction Accuracy

**Mathematical Formulation**:
```
Accuracy = (1/N) ∑_{i=1}^N I(||a_pred_i - a_true_i|| < ε)
```

Where:
- `I(·)`: Indicator function
- `ε`: Tolerance threshold

---

## 13. Convergence Analysis

### 13.1 Convergence Conditions

**Theorem**: CALM converges if:
1. Loss functions are Lipschitz continuous
2. Learning rate satisfies Robbins-Monro conditions
3. Batch size is sufficiently large

**Robbins-Monro Conditions**:
```
∑ η_t = ∞
∑ η_t² < ∞
```

### 13.2 Convergence Rate

**Expected Rate**: O(1/√T) under standard conditions
- T: Number of training iterations

---

## 14. Extensions

### 14.1 Multi-Head Curiosity

**Mathematical Formulation**:
```
C_multi = ∑_{k=1}^K α_k * C_k(s_t, a_t, s_{t+1})
```

Where:
- `C_k`: k-th curiosity head
- `α_k`: Weight for k-th head

### 14.2 Hierarchical Alignment

**Mathematical Formulation**:
```
L_align = ∑_{l=1}^L λ_l * ||f_lang^l(text) - f_action^l(action)||²
```

Where:
- `l`: Layer index
- `λ_l`: Weight for layer l

---

## 15. Summary of Key Equations

| Component | Equation | Description |
|-----------|----------|-------------|
| Curiosity | `C = D_KL[q(z\|o,h) \|\| p(z\|h)]` | KL divergence information gain |
| Forward Model | `s_{t+1} = f(s_t, a_t) + ε` | Next state prediction |
| Alignment Loss | `L_align = \|f_lang(text) - f_action(action)\|²` | Action-language alignment |
| Total Loss | `L_total = L_lm + λL_align + γL_curiosity + δL_forward` | Combined loss |
| Motor Entropy | `L_entropy = -α * H(π(a\|h))` | Encourage diverse actions |
| Generation | `w_{t+1} = argmax [P(w\|w_{<t}) + β * C(w_{<t}, w)]` | Curiosity-driven generation |

---

## 16. References

1. **Information Theory**: Shannon, C. E. (1948). "A Mathematical Theory of Communication"
2. **KL Divergence**: Kullback, S., & Leibler, R. A. (1951). "On Information and Sufficiency"
3. **Transformer**: Vaswani et al. (2017). "Attention Is All You Need"
4. **Curiosity**: Pathak et al. (2017). "Curiosity-Driven Exploration by Self-Supervised Prediction"
5. **Adam**: Kingma & Ba (2014). "Adam: A Method for Stochastic Optimization"
