# Modular Addition Transformer - Mechanistic Interpretability

## Project Overview

This project studies **grokking** in modular addition transformers using mechanistic interpretability tools (TransformerLens, devinterp). It contains multiple Jupyter notebooks exploring different aspects of the problem.

## Notebooks

- `grokking-transformerlens-explained.ipynb` - Main TransformerLens analysis with detailed explanations
- `grokking-transformerlens-fixed.ipynb` - Fixed version of TransformerLens analysis
- `grokking-transformerlens-original.ipynb` - Original TransformerLens notebook
- `grokking-original.ipynb` - Original grokking training notebook
- `grokking-devinterp.ipynb` - Using devinterp library 

## Connecting to Jupyter MCP Server

The Jupyter MCP server is configured in Claude Code settings. To connect and work with notebooks:

### 1. Find or start the Jupyter server

```bash
# Check for running servers (use the local .venv)
.venv/bin/jupyter server list

# If no server is running, start one:
.venv/bin/jupyter lab --no-browser --port=8888 --ServerApp.token='mcp' --ip=127.0.0.1
```

### 2. Connect MCP to the Jupyter server

Use the `connect_to_jupyter` MCP tool:
- URL: `http://localhost:8888`
- Token: whatever token the server reports (check `jupyter server list` output)

### 3. Use a notebook

Use the `use_notebook` MCP tool:
- `notebook_name`: a short identifier (e.g., `grokking-transformerlens-explained`)
- `notebook_path`: path **relative to Jupyter server root**, NOT absolute (e.g., `grokking-transformerlens-explained.ipynb`)
- `mode`: `connect` for existing notebooks, `create` for new ones

**Important**: The Jupyter server root is the directory from which `jupyter lab` was launched. Use `list_files` MCP tool to see the file tree from Jupyter's perspective and find the correct relative path.

### 4. Common MCP operations

- `read_notebook` - Read cell contents (use `brief` format for overview, `detailed` for full source)
- `read_cell` - Read a specific cell with outputs
- `execute_cell` - Run a specific cell
- `insert_execute_code_cell` - Insert and run a new code cell
- `overwrite_cell_source` - Edit an existing cell
- `execute_code` - Run code in kernel without saving to notebook (good for quick checks, `print()`, magic commands)

# Grokking on Modular Addition in Mechanistic Interpretability

Training a 1-layer transformer on modular addition (a + b mod 113) and analyzing the grokking phenomenon, where the model first memorizes the training data, then suddenly generalizes to the test set after continued training.

Mutating code from Neel Nanda, TransformerLens, devinterp, and others. https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Grokking_Demo.ipynb , https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/grokking.ipynb , https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Grokking_Analysis.ipynb 
 
## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Model Architecture

- 1-layer transformer with 4 attention heads
- `d_model=128`, `d_head=32`, `d_mlp=512`
- ReLU activation, no normalization, biases disabled
- Vocabulary: 114 tokens (0–112 for numbers, 113 for `=`)
- Input format: `[a, b, =]` → predicts `(a + b) mod 113`

## Pretrained Model

The trained model checkpoint is available on Hugging Face: [BurnyCoder/grokking-modular-addition-transformer](https://huggingface.co/BurnyCoder/grokking-modular-addition-transformer)

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
