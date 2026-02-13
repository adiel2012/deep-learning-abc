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

## BERT Family & Fine-Tuning

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/bert_family.ipynb) **bert_family.ipynb** | Mid-level | Comprehensive implementation of **BERT**, **RoBERTa** (no NSP, dynamic mask), **ALBERT** (param sharing), **DistilBERT** (distillation), and **DeBERTa** (disentangled attention). |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/bert_fine_tuning.ipynb) **bert_fine_tuning.ipynb** | High-level | Fine-tuning BERT for Sequence Classification, Token Classification (NER), and Question Answering (SQuAD). |

### GPT Family & Generative Models

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/gpt2.ipynb) **gpt2.ipynb** | Mid-level | **GPT-2** from scratch. Causal masked attention, pre-norm architecture, and text generation loop. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/gpt3.ipynb) **gpt3.ipynb** | Mid-level | **GPT-3** architecture (scale & patterns) and the concept of few-shot in-context learning. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/t5.ipynb) **t5.ipynb** | Mid-level | **T5** (Text-to-Text Transfer Transformer). Encoder-decoder architecture with relative positional bias and unified task format. |

### Advanced Transformer Architectures

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/xlnet.ipynb) **xlnet.ipynb** | Advanced | **XLNet**: Permutation Language Modeling and Two-Stream Attention mechanism suitable for autoregressive bidirectional context. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/transformer_xl.ipynb) **transformer_xl.ipynb** | Advanced | **Transformer-XL**: Segment-level recurrence and relative positional encodings for long-term dependency modeling. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/longformer.ipynb) **longformer.ipynb** | Advanced | **Longformer**: Sparse attention mechanism combining sliding window, dilated window, and global attention for long sequences. |

### Computer Vision

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/vision_transformer.ipynb) **vision_transformer.ipynb** | Mid-level | **Vision Transformer (ViT)**. Patch embeddings, CLS token for classification, and position embeddings applied to images. |

### Generative AI

| Notebook | Level | Description |
|----------|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/gan.ipynb) **gan.ipynb** | Mid-level | **GAN** (Generative Adversarial Network). Minimax game between Generator and Discriminator. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/deep-learning-abc/blob/main/vae.ipynb) **vae.ipynb** | Mid-level | **VAE** (Variational Autoencoder). Evidence Lower Bound (ELBO) maximization, reparameterization trick, and latent space sampling. |

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

### Model Evolution: Then vs Now

| Feature | [BERT](bert_family.ipynb) (2018) | [GPT-2](gpt2.ipynb) (2019) | [T5](t5.ipynb) (2020) | [ViT](vision_transformer.ipynb) (2020) | [Longformer](longformer.ipynb)/[XL](transformer_xl.ipynb) (2019/20) |
|---------|-------------|--------------|-----------|------------|-------------------------|
| **Architecture** | Encoder-only | Decoder-only | Encoder-Decoder | Encoder-only (on patches) | Encoder-only (Recurrent/Sparse) |
| **Objective** | MLM + NSP | Causal LM | Span Corruption (Text-to-Text) | Supervised Classif. | Autoregressive / MLM |
| **Attention** | Bidirectional Full | Causal Full | Bidirectional Enc / Causal Dec | Bidirectional Full | Recurrent / Sparse |
| **Positional Encoding** | Learned Absolute | Learned Absolute | Relative Bias (Bucketed) | Learned Absolute | Relative (Sinusoidal) |
| **Normalization** | Post-LN | Pre-LN | Pre-LN (No bias) | Pre-LN | Pre-LN |
| **Context** | 512 | 1024 | 512 | Fixed by patch count | Extended / 4096+ |

## Running Locally

```bash
pip install torch matplotlib
jupyter notebook
```

## Running with Docker

To ensure a consistent environment, you can run the notebooks using Docker:

```bash
# Build the image
docker build -t dl-notebooks .

# Run the container
docker run -p 8888:8888 dl-notebooks
```

Then open the link displayed in the terminal (usually `http://127.0.0.1:8888/?token=...`).

Or click any Colab badge above to run in your browser with free GPU access.
