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

## Running Locally

```bash
pip install torch matplotlib
jupyter notebook
```

Or click any Colab badge above to run in your browser with free GPU access.
