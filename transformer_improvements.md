# Extensive List of Improvements to Transformers Since "Attention Is All You Need"

Since the release of "Attention Is All You Need" (Vaswani et al., 2017), the Transformer architecture has undergone radical evolution. While the core scaled dot-product attention mechanism remains foundational, nearly every other component has been optimized for longer contexts, training stability, computational efficiency, and hardware utilization.

---

## 1. Positional Encoding Improvements

The original Transformer used fixed sinusoidal functions to encode token positions. This was one of the first components to evolve.

### Relative Positional Encodings (Shaw et al., 2018)
- **Key Innovation**: Encodes the *distance* between tokens rather than absolute positions
- **Benefits**: Better generalization to unseen sequence lengths
- **Used in**: Transformer-XL, T5

### RoPE (Rotary Positional Embeddings) (Su et al., 2021)
- **Key Innovation**: Rotates Query and Key vectors in high-dimensional space
- **Benefits**: 
  - Natural decay as token distance increases
  - Excellent extrapolation to longer contexts
  - No added parameters
- **Used in**: LLaMA, PaLM, GPT-NeoX, Mistral, Gemma
- **Status**: Current industry standard

### ALiBi (Attention with Linear Biases) (Press et al., 2022)
- **Key Innovation**: Adds distance-based penalty directly to attention scores (no positional embeddings)
- **Benefits**: 
  - Extreme length extrapolation (trained on 2K → inference on 32K+)
  - Zero learnable parameters for positions
- **Used in**: BLOOM, MPT

### Learned Positional Embeddings
- **Key Innovation**: Treat positions as learnable parameters
- **Benefits**: Maximum flexibility for training data
- **Drawback**: Poor extrapolation beyond training length
- **Used in**: BERT, GPT-2

### xPos (Sun et al., 2022)
- **Key Innovation**: Exponential decay applied to RoPE for better long-range modeling
- **Used in**: Some variants of LLaMA

---

## 2. Attention Mechanism Variants

Standard self-attention has O(n²) complexity—doubling sequence length quadruples compute. These improvements address this bottleneck.

### FlashAttention (Dao et al., 2022-2024)
- **Versions**: FlashAttention 1, 2, and 3
- **Key Innovation**: IO-aware algorithm optimizing GPU memory movement (SRAM ↔ HBM)
- **Benefits**:
  - 2-4× faster training
  - Enables much longer contexts with same memory
  - Mathematically identical to standard attention
- **Status**: De facto standard for modern training

### Multi-Query Attention (MQA) (Shazeer, 2019)
- **Key Innovation**: All attention heads share the same Key and Value projections
- **Benefits**:
  - Dramatically reduced KV cache during inference
  - 4-8× faster autoregressive generation
- **Drawback**: Slight quality degradation
- **Used in**: PaLM, Falcon

### Grouped-Query Attention (GQA) (Ainslie et al., 2023)
- **Key Innovation**: Compromise between MHA and MQA—groups of heads share KV projections
- **Benefits**: 90-95% of MQA speed with nearly full MHA quality
- **Used in**: LLaMA 2, LLaMA 3, Mistral, Gemma
- **Status**: Current best practice for production models

### Sliding Window Attention (Beltagy et al., 2020; Child et al., 2019)
- **Key Innovation**: Each token attends only to a fixed window of previous tokens
- **Benefits**: 
  - O(n) complexity instead of O(n²)
  - Through layer stacking, achieves large receptive field
- **Used in**: Longformer, Mistral 7B
- **Typical window size**: 512-4096 tokens

### Linear Attention Variants
Multiple approaches to approximate attention with linear complexity:

#### Linformer (Wang et al., 2020)
- Projects keys/values to lower dimension
- O(n) complexity

#### Performer (Choromanski et al., 2021)
- Uses random feature approximations (kernel methods)
- True O(n) with quality tradeoffs

#### RWKV (Peng et al., 2023)
- Reformulates attention as an RNN-like recurrence
- Constant memory during inference

### Multi-Head Latent Attention (MLA)
- **Key Innovation**: Used in DeepSeek-V2; reduces KV cache by compressing into latent space
- **Benefits**: Extreme efficiency for inference

### Cross-Attention Optimizations
- **Perceiver** (Jaegle et al., 2021): Fixed latent array attends to inputs
- **Perceiver IO**: Extends to outputs

---

## 3. Normalization and Training Stability

The original Post-LayerNorm was notoriously unstable and required careful warm-up schedules.

### Pre-LayerNorm (Pre-LN)
- **Key Innovation**: Apply LayerNorm *before* attention/FFN rather than after
- **Benefits**:
  - Dramatically more stable training
  - Enables training 100+ layer models
  - Reduces need for warm-up
- **Used in**: GPT-2+, LLaMA, nearly all modern LLMs
- **Status**: Current standard

### RMSNorm (Zhang & Sennrich, 2019)
- **Key Innovation**: Only normalizes by root mean square (no mean centering)
- **Benefits**:
  - 10-30% faster than LayerNorm
  - Equivalent or better performance
  - Simpler implementation
- **Used in**: LLaMA, T5, Gopher, Chinchilla
- **Status**: Preferred in modern LLMs

### DeepNorm (Wang et al., 2022)
- **Key Innovation**: Scales residual connections to stabilize extremely deep models
- **Benefits**: Enables training up to 1,000 layers
- **Used in**: DeepNet experiments, some research models

### QK-Norm (Dehghani et al., 2023)
- **Key Innovation**: Normalize Query and Key separately before attention
- **Benefits**: Prevents attention logit growth in very large models

### LayerScale (Touvron et al., 2021)
- **Key Innovation**: Learnable scaling parameter for each residual connection
- **Benefits**: Improves training of deep ViTs and some language models

---

## 4. Feed-Forward Networks (FFN) and Activation Functions

The FFN comprises ~67% of Transformer parameters and is critical for model capacity.

### GeLU (Gaussian Error Linear Unit) (Hendrycks & Gimpel, 2016)
- **Key Innovation**: Smooth, probabilistic activation function
- **Benefits**: Better gradient flow than ReLU
- **Used in**: BERT, GPT-2, GPT-3
- **Formula**: `x * Φ(x)` where Φ is the cumulative distribution function

### SwiGLU (Shazeer, 2020)
- **Key Innovation**: Gated Linear Unit with Swish activation
- **Benefits**:
  - More expressive than GeLU/ReLU
  - Better performance for same parameter count
- **Used in**: PaLM, LLaMA, LLaMA 2/3, Mistral, Gemma
- **Status**: Gold standard for modern LLMs
- **Formula**: `Swish(xW) ⊗ (xV)` where ⊗ is element-wise multiplication

### GeGLU
- Variant using GeLU instead of Swish
- Used in some T5 variants

### Sparse FFN Layers
- **Key Innovation**: Only activate a subset of FFN neurons per token
- Early work leading to Mixture of Experts

---

## 5. Mixture of Experts (MoE)

Revolutionary approach to scaling models without proportional compute increases.

### Switch Transformer (Fedus et al., 2021)
- **Key Innovation**: Replace dense FFN with many expert networks; route each token to one expert
- **Benefits**: 
  - 7× more efficient than dense models
  - Can train trillion-parameter models
- **Routing**: Top-1 expert selection

### GLaM (Du et al., 2021)
- **Key Innovation**: Improved routing with top-2 experts per token
- Used by Google for efficient scaling

### Mixtral 8x7B (Mistral AI, 2023)
- **Key Innovation**: 8 experts, activate 2 per token
- **Benefits**: 
  - 47B total parameters, 13B active per token
  - Matches or beats much larger dense models
- **Status**: Demonstrates production viability of MoE

### DeepSeek-MoE (DeepSeek, 2024)
- **Key Innovation**: Fine-grained expert segmentation with shared experts
- Achieves better expert utilization

### Expert Capacity and Load Balancing
- **Challenge**: Uneven expert usage causes training instability
- **Solutions**: 
  - Auxiliary loss terms to encourage balanced routing
  - Expert capacity limits
  - Token dropping mechanisms

---

## 6. Long Context Extensions

Extending context length beyond training has been a major research focus.

### Context Scaling Techniques

#### Position Interpolation (Chen et al., 2023)
- **Key Innovation**: Compress positional encodings to fit longer sequences
- **Example**: LLaMA trained on 2K → extended to 32K
- Used with RoPE

#### YaRN (Yet another RoPE extensioN) (Peng et al., 2023)
- **Key Innovation**: Interpolates low frequencies, extrapolates high frequencies
- Better preservation of model capabilities

#### NTK-Aware Scaling
- Adjusts RoPE's base frequency for better extrapolation
- Simple but effective

### Recurrent Memory Mechanisms

#### Transformer-XL (Dai et al., 2019)
- **Key Innovation**: Caches previous segment representations
- Effective context length grows linearly with depth

#### Compressive Transformer (Rae et al., 2020)
- Compresses old memories into smaller representations

### Landmark Attention (Mohtashami & Jaggi, 2023)
- Blocks of tokens attend to "landmark" tokens for efficiency

---

## 7. Training Efficiency and Optimization

### Optimizer Improvements

#### AdamW (Loshchilov & Hutter, 2019)
- Decoupled weight decay from gradient updates
- Standard optimizer for Transformers

#### Lion (Chen et al., 2023)
- More memory-efficient than Adam
- Competitive or better performance

#### Sophia (Liu et al., 2023)
- Second-order optimizer using Hessian information
- 2× faster convergence for LLMs

### Learning Rate Schedules

#### Cosine Decay with Warm-up
- Current standard for LLM training
- Smooth decay prevents training collapse

#### Inverse Square Root Decay
- Used in original Transformer and T5
- Good for continuous training

### Mixed Precision Training

#### FP16 (Half Precision)
- Standard since 2018
- 2× memory savings, faster training

#### BF16 (Brain Float16)
- Better numerical stability than FP16
- Standard on modern GPUs (A100, H100)

#### FP8 Training (2023+)
- Emerging standard for H100 GPUs
- Further 2× speedup

### Gradient Checkpointing (Activation Recomputation)
- Trade compute for memory
- Enables training much larger models

---

## 8. Initialization and Scaling Laws

### Scaling Law Discoveries (Kaplan et al., 2020; Hoffmann et al., 2022)

#### Chinchilla Scaling Laws
- **Key Finding**: Most models are undertrained
- **Recommendation**: For compute budget C, optimal is ~20 tokens per parameter
- Revolutionized training strategies

### Initialization Improvements

#### μP (Maximal Update Parametrization) (Yang et al., 2022)
- **Key Innovation**: Hyperparameters transfer across model scales
- Enables cheaper hyperparameter tuning on small models

#### Small Init (for Post-LN)
- Scale initial weights by 1/√depth
- Prevents gradient explosion

---

## 9. Architectural Variants and Hybrids

### Encoder-Decoder vs. Decoder-Only

#### Decoder-Only Dominance
- **Trend**: GPT-style decoder-only has become dominant for LLMs
- **Reason**: Simpler, scales better, autoregressive pretraining is more flexible
- **Examples**: GPT series, LLaMA, Mistral, Gemini

#### Encoder-Only
- **Used for**: Classification, embeddings (BERT, RoBERTa)
- Less common for generative tasks

#### Encoder-Decoder Persistence
- **Used for**: Translation, summarization tasks with clear input/output separation
- **Examples**: T5, BART, mT5

### Hybrid Architectures

#### RetNet (Sun et al., 2023)
- Combines advantages of Transformers and RNNs
- O(1) inference complexity

#### Mamba (Gu & Dao, 2023)
- State Space Model (SSM) that rivals Transformers
- O(n) complexity, strong performance

#### Griffin (DeepMind, 2024)
- Alternates local attention with gated recurrent layers

---

## 10. Embedding and Vocabulary Improvements

### Tokenization Advances

#### SentencePiece (Kudo & Richardson, 2018)
- Unsupervised tokenization
- Language-agnostic
- Standard for multilingual models

#### Byte-Pair Encoding (BPE) Variants
- Original GPT-2 used BPE
- Improvements handle whitespace and rare words better

#### Byte-Level Tokenization
- Used in GPT-4
- Completely language-agnostic
- Can represent any text

### Embedding Techniques

#### Rotary Embeddings for Tokens
- Apply RoPE principles to token embeddings

#### Learnable Embedding Initialization
- Better than random initialization
- Especially for domain-specific vocabularies

---

## 11. Model Compression and Efficiency

### Quantization

#### Post-Training Quantization (PTQ)
- **GPTQ** (Frantar et al., 2023): 4-bit quantization with minimal degradation
- **AWQ** (Lin et al., 2023): Activation-aware weight quantization

#### Quantization-Aware Training (QAT)
- Train with quantization in mind
- Better quality than PTQ

### Pruning and Distillation

#### Structured Pruning
- Remove entire attention heads or FFN dimensions
- Maintains efficiency on hardware

#### Knowledge Distillation
- Train smaller "student" model to mimic larger "teacher"
- **DistilBERT**, **TinyBERT**: Successful compressed models

### Low-Rank Adaptations

#### LoRA (Hu et al., 2021)
- **Key Innovation**: Fine-tune low-rank adapter matrices instead of full weights
- **Benefits**: 
  - 1000× fewer trainable parameters
  - 3× faster training
  - Easily swap adapters for different tasks

#### QLoRA (Dettmers et al., 2023)
- Combines LoRA with 4-bit quantization
- Fine-tune 65B models on single GPU

---

## 12. Parallelism Strategies

Modern LLMs require distributed training across hundreds or thousands of GPUs.

### Tensor Parallelism
- Split individual weight matrices across GPUs
- Used within a single node

### Pipeline Parallelism
- Split model layers across GPUs
- Enables training very deep models

### Sequence Parallelism
- Split sequence dimension across GPUs
- Reduces activation memory

### Data Parallelism
- Replicate model, split data batches
- Most common parallelism strategy

### ZeRO (Zero Redundancy Optimizer) (Rajbhandari et al., 2020)
- Eliminates memory redundancy in data parallelism
- Enables training 100B+ parameter models

### FSDP (Fully Sharded Data Parallel)
- PyTorch implementation of ZeRO concepts
- Standard for distributed training

---

## 13. Attention Masking Improvements

### Causal Masking Optimizations
- Efficient implementation of autoregressive masking
- Fused with attention kernels (FlashAttention)

### Prefix LM
- Attend bidirectionally to prefix, causally to suffix
- Useful for instruction-following

### Sparse Attention Patterns

#### Blockwise Attention
- Used in Sparse Transformers

#### Axial Attention
- Separate attention along different axes
- Efficient for 2D/3D inputs (images, video)

---

## 14. Loss Function and Training Objectives

### z-Loss (Chowdhery et al., 2022)
- Auxiliary loss to prevent logit drift
- Used in PaLM

### Next Token Prediction Improvements
- Better sampling strategies
- Importance sampling for rare tokens

### Contrastive Learning Integration
- SimCLR-style objectives for embeddings
- Used in some multimodal models

---

## 15. Specialized Modules

### Retrieval-Augmented Components

#### RETRO (Borgeaud et al., 2022)
- Retrieves from external database during attention
- Dramatically reduces parameters needed for same performance

#### kNN-LM
- Nearest-neighbor search over training data
- Improves perplexity without model changes

### Adapter Layers
- Small trainable modules inserted into frozen models
- Enables parameter-efficient fine-tuning

### Memory Networks
- External key-value memory
- Used in some reasoning models

---

## 16. Summary: Then vs. Now

| Component | Original (2017) | Modern (2024-2025) |
|-----------|----------------|-------------------|
| **Positional Encoding** | Fixed sinusoidal (absolute) | RoPE or ALiBi (relative) |
| **Normalization** | Post-LayerNorm | Pre-RMSNorm |
| **Activation** | ReLU | SwiGLU |
| **Attention** | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) + FlashAttention |
| **Complexity** | O(n²) | O(n²) but 10× faster (FlashAttention) or O(n) (Sliding Window) |
| **Architecture** | Dense | Dense or Sparse (MoE) |
| **Context Length** | 512 tokens | 8K-128K+ tokens (routinely) |
| **Optimizer** | Adam | AdamW, Lion, Sophia |
| **Precision** | FP32 | BF16, FP8 |
| **Parallelism** | Data parallelism | ZeRO, FSDP, Tensor/Pipeline parallelism |
| **Fine-tuning** | Full model | LoRA, QLoRA adapters |

---

## 17. Key Research Papers Timeline

### 2017
- **Attention Is All You Need** (Vaswani et al.) - The original Transformer

### 2018
- **BERT** (Devlin et al.) - Bidirectional pretraining
- **GPT** (Radford et al.) - Autoregressive language modeling
- **Self-Attention with Relative Position Representations** (Shaw et al.)

### 2019
- **Transformer-XL** (Dai et al.) - Recurrent memory for long context
- **RoBERTa** (Liu et al.) - Improved BERT training
- **ALBERT** (Lan et al.) - Parameter sharing
- **Multi-Query Attention** (Shazeer) - KV cache reduction

### 2020
- **GPT-3** (Brown et al.) - Scaling laws demonstration
- **Longformer** (Beltagy et al.) - Sparse attention patterns
- **Linformer** (Wang et al.) - Linear complexity approximation
- **SwiGLU** (Shazeer) - Improved activation function
- **Scaling Laws** (Kaplan et al.) - Parameter/compute relationships

### 2021
- **Switch Transformer** (Fedus et al.) - Mixture of Experts
- **RoPE** (Su et al.) - Rotary positional embeddings
- **LoRA** (Hu et al.) - Low-rank adaptation
- **CoT Prompting** (Wei et al.) - Chain-of-thought reasoning

### 2022
- **Chinchilla** (Hoffmann et al.) - Revised scaling laws
- **FlashAttention** (Dao et al.) - IO-aware attention
- **ALiBi** (Press et al.) - Linear bias positional encoding
- **PaLM** (Chowdhery et al.) - Scaling to 540B parameters
- **LLaMA** (Touvron et al.) - Open efficient models

### 2023
- **GPT-4** - Multimodal capabilities (rumored MoE)
- **LLaMA 2** (Touvron et al.) - Improved training and safety
- **Mistral 7B** - Sliding window + GQA
- **Mixtral 8x7B** - Production MoE model
- **FlashAttention-2** - Further optimizations
- **QLoRA** (Dettmers et al.) - Quantized LoRA
- **Mamba** (Gu & Dao) - SSM alternative to attention
- **Grouped-Query Attention** (Ainslie et al.) - MHA/MQA compromise

### 2024-2025
- **LLaMA 3** - Further scaling and improvements
- **Gemini** - Advanced multimodal capabilities
- **FlashAttention-3** - H100 optimizations
- **DeepSeek-V2** - Multi-head latent attention + MoE innovations

---

## 18. Future Directions and Open Problems

### Active Research Areas

1. **Sub-Quadratic Attention**: Finding better O(n log n) or O(n) approximations without quality loss
2. **Infinite Context**: Truly unbounded context windows
3. **Sample Efficiency**: Better learning from less data
4. **Multimodal Integration**: Seamless unified architectures for text/image/video/audio
5. **Continual Learning**: Models that learn without catastrophic forgetting
6. **Interpretability**: Understanding what models learn and how they reason
7. **Energy Efficiency**: Reducing carbon footprint of training and inference
8. **Architecture Search**: Automated discovery of better architectures

### Emerging Paradigms

- **Test-Time Compute**: Using more compute during inference (o1-style reasoning)
- **Retrieval Augmentation**: Hybrid parametric/non-parametric memory
- **Neurosymbolic Integration**: Combining neural networks with symbolic reasoning
- **Smaller Specialized Models**: Moving away from "bigger is always better"

---

## Conclusion

The Transformer architecture has evolved from a novel attention-based sequence-to-sequence model into the foundation of modern AI. While the core attention mechanism remains, virtually every other component has been optimized, replaced, or augmented. Key trends include:

1. **Efficiency First**: FlashAttention, GQA, MoE enable dramatically more capable models
2. **Longer Contexts**: From 512 to 128K+ tokens through RoPE, ALiBi, and clever engineering
3. **Training Stability**: Pre-LN and RMSNorm make deep models trainable
4. **Better Scaling**: Chinchilla laws and improved optimizers maximize compute efficiency
5. **Production Readiness**: Quantization, LoRA, and inference optimizations democratize access

The Transformer's modularity has been its greatest strength, allowing researchers to systematically improve each component. The architecture continues to evolve rapidly, with new improvements announced monthly. What began as "attention is all you need" has become "attention plus lots of clever engineering is all you need."
