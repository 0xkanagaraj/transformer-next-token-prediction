# Transformer-Based Next-Token Prediction with Basic Self-Attention

A small decoder-only Transformer built **from scratch in PyTorch** that learns to predict the next character in a text. Self-attention is implemented by hand (queries, keys, values, scaled dot-product scores, causal mask and softmax) rather than with a built-in attention layer, so the mechanics behind GPT-style language models are easy to follow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0xkanagaraj/transformer-next-token-prediction/blob/main/transformer_next_token.ipynb)

## Overview

Language models are trained on one simple task: given the tokens so far, predict the next one. This project trains a character-level model on Shakespeare text and covers the full workflow: tokenization, a hand-written self-attention head, multi-head attention, Transformer blocks, training, evaluation, text generation and attention visualization.

## Dataset

The notebook downloads the **Tiny Shakespeare** corpus (about 1 MB of plain text, roughly 1.1 million characters) from the `karpathy/char-rnn` GitHub repository. The text is split 90% train and 10% validation.

If the download fails, the notebook falls back to a very small built-in passage so it still runs. Results will be much weaker in that case.

## Repository Structure

```
transformer-next-token-prediction/
├── transformer_next_token.ipynb   # Colab notebook with the full code
└── README.md
```

## Getting Started

### Run in Google Colab

1. Click the **Open In Colab** badge above.
2. Select **Runtime → Change runtime type → GPU** (T4) for faster training.
3. Choose **Runtime → Run all**.

### Run locally

```bash
git clone https://github.com/0xkanagaraj/transformer-next-token-prediction.git
cd transformer-next-token-prediction
pip install torch numpy matplotlib jupyter
jupyter notebook transformer_next_token.ipynb
```

## How It Works

**Self-attention head.** Every token is projected into a query, a key and a value. Attention scores are `softmax(QKᵀ / √d)`, and a causal (lower-triangular) mask stops each token from seeing future tokens. The output is the attention-weighted sum of the values.

**Transformer block.** Multi-head attention followed by a feed-forward network, each wrapped with layer normalization and a residual connection.

**Model.** Token embeddings plus learned positional embeddings feed a stack of blocks, then a final layer norm and a linear layer that outputs a score for every character in the vocabulary. Training minimizes cross-entropy between the predicted and actual next character.

## Model Configuration

| Setting | Value |
|---------|-------|
| Tokenization | Character level |
| Context length | 128 |
| Embedding size | 128 |
| Attention heads | 4 per layer |
| Transformer blocks | 4 |
| Dropout | 0.1 |
| Optimizer | AdamW (learning rate 3e-4) |
| Batch size | 64 |
| Training steps | 3,000 |

The model has roughly 0.8 million parameters.

## Workflow

1. Load the text and build a character vocabulary
2. Create training batches where the target is the input shifted by one character
3. Build the self-attention head, multi-head attention, feed-forward network and Transformer blocks
4. Train with AdamW and gradient clipping, tracking train and validation loss
5. Report validation loss, perplexity and top-1 next-token accuracy
6. Show the top-5 predicted next characters for a prompt
7. Generate text with temperature and top-k sampling
8. Plot attention maps to see what the model looks at
9. Save the trained weights

## Requirements

- Python 3.8+
- torch
- numpy
- matplotlib

All of these are preinstalled in Google Colab.

## Results

Add your final numbers and plots here after running the notebook, for example:

| Metric | Value |
|--------|-------|
| Validation loss | _your value_ |
| Validation perplexity | _your value_ |
| Next-token accuracy | _your value_ |

You can also paste a short generated sample and the attention heatmap.

## Possible Extensions

- Switch to subword tokenization (BPE) instead of characters
- Train on a different corpus (code, song titles, your own writing)
- Add rotary or sinusoidal positional encodings
- Compare the number of heads, layers and context length
- Use `torch.nn.functional.scaled_dot_product_attention` for speed
- Scale up with a larger model and longer training

## Author

[0xkanagaraj](https://github.com/0xkanagaraj)
