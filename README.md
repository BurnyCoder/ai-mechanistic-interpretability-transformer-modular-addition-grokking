# Grokking on Modular Addition — Mechanistic Interpretability

Training a 1-layer transformer on modular addition (a + b mod 113) and analyzing the grokking phenomenon — where the model first memorizes the training data, then suddenly generalizes to the test set after continued training.

Mutating code from Neel Nanda, TransformerLens, devinterp, and others.

## Model Architecture

- 1-layer transformer with 4 attention heads
- `d_model=128`, `d_head=32`, `d_mlp=512`
- ReLU activation, no normalization, biases disabled
- Vocabulary: 114 tokens (0–112 for numbers, 113 for `=`)
- Input format: `[a, b, =]` → predicts `(a + b) mod 113`

## Training Setup

- 30% of all p² = 12,769 input pairs used for training
- Full-batch training with AdamW (lr=1e-3, weight decay=1.0)
- 25,000 epochs with checkpoints every 100 epochs

## Analysis

The notebook (`grokking-transformerlens.ipynb`) performs mechanistic interpretability analysis:

- **Fourier analysis of embeddings**: The model learns embeddings sparse in the Fourier basis, concentrating on key frequencies (17, 25, 32, 47)
- **Neuron activation patterns**: MLP neurons cluster by frequency, each responding to specific Fourier components of `(a + b)`
- **Attention patterns**: How attention heads route information from input tokens to the output position
- **SVD of weight matrices**: Principal components of the embedding and neuron-to-logit weight matrices
- **Logit periodicity**: Output logits are well-approximated by `cos(freq * 2π/p * (a + b - c))` for the key frequencies
- **Progress measures**: Restricted loss and excluded loss across training reveal three phases:
  1. **Memorization** (~epoch 0–1500)
  2. **Circuit formation** (~epoch 1500–13300)
  3. **Cleanup** (~epoch 13300–16600)

## Algorithm Learned by the Model

The transformer learns a Fourier-based algorithm:
1. **Embed** inputs `a` and `b` into Fourier components (sin/cos at key frequencies)
2. **Attend** from the `=` position to `a` and `b`, computing representations of `sin(ωa)`, `cos(ωa)`, `sin(ωb)`, `cos(ωb)`
3. **MLP neurons** compute `cos(ω(a+b))` and `sin(ω(a+b))` via trig identities
4. **Unembed** maps these to logits approximating `cos(ω(a+b-c))` for each output token `c`

## References

- [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217) (Nanda et al., 2023)
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) (Power et al., 2022)
