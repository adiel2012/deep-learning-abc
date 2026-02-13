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

## Running Locally

```bash
pip install torch matplotlib
jupyter notebook
```

Or click any Colab badge above to run in your browser with free GPU access.
