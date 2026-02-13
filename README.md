# deep-learning-abc

Hands-on notebooks that implement deep learning concepts from scratch. Each notebook is self-contained and runs directly in Google Colab — no local setup required.

## Notebooks

### Attention Mechanism

Two complementary notebooks covering the same topic at different abstraction levels:

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/attention_from_scratch.ipynb) **attention_from_scratch.ipynb** | Low-level | Raw tensor ops only — no `nn.Module`. Includes math foundations (dot products, softmax, scaling). |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/attention_with_pytorch.ipynb) **attention_with_pytorch.ipynb** | Mid-level | Uses `nn.Module`, `nn.Linear`, `F.softmax`. Includes a training loop, parameter inspection, and comparison with `nn.MultiheadAttention`. |

**Topics covered:** scaled dot-product attention, causal masking, multi-head attention (loop and batched versions), attention weight visualization.

### Positional Encoding

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/positional_encoding.ipynb) **positional_encoding.ipynb** | Low-level | Why attention needs position info, sinusoidal encoding math & implementation, learned embeddings, visualizations. |

**Topics covered:** permutation invariance problem, sinusoidal PE formula & properties, frequency spectrum intuition, relative position via dot product, learned vs fixed embeddings, scaling with √d_model.

### Transformer Encoder

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/transformer_encoder.ipynb) **transformer_encoder.ipynb** | Low-level | Full encoder from "Attention Is All You Need" built from raw ops. Layer norm, residual connections, FFN, stacked layers. |

**Topics covered:** residual connections (gradient highway), layer normalization (vs batch norm), position-wise FFN, encoder layer assembly, N-layer stacking, parameter counting, attention visualization across layers, representation evolution, residual vs no-residual comparison.

---

## Transformer Improvements (Since 2017)

The notebooks below implement the key improvements to the Transformer architecture since the original "Attention Is All You Need" paper.

### Positional Encoding Improvements: RoPE & ALiBi

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/rope_alibi.ipynb) **rope_alibi.ipynb** | Low-level | RoPE (rotary embeddings) and ALiBi (linear biases) implemented from scratch. Comparison with sinusoidal PE, length extrapolation analysis. |

**Topics covered:** RoPE rotation matrices, frequency computation, relative position via dot product, ALiBi distance penalties, per-head slopes, attention entropy comparison, length extrapolation.

### Attention Variants: MQA, GQA & Sliding Window

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/attention_variants.ipynb) **attention_variants.ipynb** | Low-level | Multi-Query, Grouped-Query, and Sliding Window attention from scratch. KV cache analysis, complexity comparison. |

**Topics covered:** MHA baseline, Multi-Query Attention (shared K,V), Grouped-Query Attention (KV groups), sliding window masking, receptive field growth through layers, KV cache scaling, O(n²) vs O(n·w) complexity.

### Normalization & Activations: Pre-LN, RMSNorm, GeLU, SwiGLU

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/normalization_activations.ipynb) **normalization_activations.ipynb** | Low-level | LayerNorm vs RMSNorm, Pre-LN vs Post-LN placement, ReLU → GeLU → SwiGLU activation progression. |

**Topics covered:** LayerNorm internals, RMSNorm (no mean centering), Post-LN instability, Pre-LN gradient highway, GeLU smooth activation, SwiGLU gating mechanism, modern LLaMA-style block.

### Mixture of Experts (MoE)

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/mixture_of_experts.ipynb) **mixture_of_experts.ipynb** | Low-level | MoE routing from scratch — top-1 (Switch Transformer) and top-2 (Mixtral) routing, load balancing loss. |

**Topics covered:** dense FFN baseline, expert routing, top-1 vs top-2 selection, router softmax, load balancing auxiliary loss, expert utilization visualization, parameter vs compute scaling analysis.

### LoRA: Low-Rank Adaptation

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/lora_fine_tuning.ipynb) **lora_fine_tuning.ipynb** | Low-level | LoRA parameter-efficient fine-tuning from scratch. Low-rank decomposition, training, adapter merging, rank analysis. |

**Topics covered:** low-rank matrix approximation (SVD), LoRA adapter initialization (B=0), scaling factor α/r, training only A and B, adapter merging for zero-overhead inference, multi-task adapter switching, parameter savings at scale.

---

### Then vs Now: Summary

| Component | Original (2017) | Modern (2024) |
|-----------|-----------------|---------------|
| Positional Encoding | Fixed sinusoidal (absolute) | RoPE or ALiBi (relative) |
| Normalization | Post-LayerNorm | Pre-RMSNorm |
| Activation | ReLU | SwiGLU |
| Attention | Multi-Head (MHA) | Grouped-Query (GQA) + FlashAttention |
| Architecture | Dense | Dense or Sparse (MoE) |
| Context Length | 512 tokens | 8K–128K+ tokens |
| Fine-tuning | Full model | LoRA / QLoRA adapters |

## Running Locally

```bash
pip install torch matplotlib
jupyter notebook
```

Or click any Colab badge above to run in your browser with free GPU access.
