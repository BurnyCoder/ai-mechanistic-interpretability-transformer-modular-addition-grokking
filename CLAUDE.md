# Modular Addition Transformer - Mechanistic Interpretability

## Project Overview

This project studies **grokking** in modular addition transformers using mechanistic interpretability tools (TransformerLens, devinterp). It contains multiple Jupyter notebooks exploring different aspects of the problem.

## Notebooks

- `grokking-transformerlens-explained.ipynb` - Main TransformerLens analysis with detailed explanations
- `grokking-transformerlens-fixed.ipynb` - Fixed version of TransformerLens analysis
- `grokking-transformerlens-original.ipynb` - Original TransformerLens notebook
- `grokking-original.ipynb` - Original grokking training notebook
- `grokking-devinterp.ipynb` - Using devinterp library 

# JupyterMCP - Jupyter Notebook Model Context Protocol Integration

JupyterMCP connects [Jupyter Notebook](https://jupyter.org/) to [Claude AI](https://claude.ai/chat) through the Model Context Protocol (MCP), allowing Claude to directly interact with and control Jupyter Notebooks. This integration enables AI-assisted code execution, data analysis, visualization, and more.

## ⚠️ Compatibility Warning

**This tool is compatible ONLY with Jupyter Notebook version 6.x.**

It does NOT work with:

- Jupyter Lab
- Jupyter Notebook v7.x
- VS Code Notebooks
- Google Colab
- Any other notebook interfaces

## Features

- **Two-way communication**: Connect Claude AI to Jupyter Notebook through a WebSocket-based server
- **Cell manipulation**: Insert, execute, and manage notebook cells
- **Notebook management**: Save notebooks and retrieve notebook information
- **Cell execution**: Run specific cells or execute all cells in a notebook
- **Output retrieval**: Get output content from executed cells with text limitation options

## Components

The system consists of three main components:

1. **WebSocket Server (`jupyter_ws_server.py`)**: Sets up a WebSocket server inside Jupyter that bridges communication between notebook and external clients
2. **Client JavaScript (`client.js`)**: Runs in the notebook to handle operations (inserting cells, executing code, etc.)
3. **MCP Server (`jupyter_mcp_server.py`)**: Implements the Model Context Protocol and connects to the WebSocket server

## Installation

### Prerequisites

- [Python 3.12 or newer](https://www.python.org/downloads/) (probably also work with older versions, but not tested)
- [`uv` package manager](/README.md#installing-uv)
- [Claude AI desktop application](https://claude.ai/download)

#### Installing uv

If you're on Mac:

```bash
brew install uv
```

On Windows (PowerShell):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For other platforms, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

1. Clone or download this repository to your computer:

   ```bash
   git clone https://github.com/jjsantos01/jupyter-notebook-mcp.git
   ```

2. Create virtual environment with required packages an install `jupyter-mcp` kernel, so it can be recognized by your jupyter installation, if you had one before.

   ```bash
   uv run python -m ipykernel install --name jupyter-mcp
   ```

3. (optional) Install additional Python packages for your analysis:

   ```bash
   uv pip install seaborn
   ```

4. Configure Claude desktop integration:
   Go to `Claude` > `Settings` > `Developer` > `Edit Config` > `claude_desktop_config.json` to include the following:

   ```json
      {
       "mcpServers": {
           "jupyter": {
               "command": "uv",
               "args": [
                   "--directory",
                   "/ABSOLUTE/PATH/TO/PARENT/REPO/FOLDER/src",
                   "run",
                   "jupyter_mcp_server.py"
               ]
           }
       }
   }
   ```

   Replace `/ABSOLUTE/PATH/TO/` with the actual path to the `src` folder on your system. For example:
   - Windows: `"C:\\Users\\MyUser\\GitHub\\jupyter-notebook-mcp\\src\\"`
   - Mac: `/Users/MyUser/GitHub/jupyter-notebook-mcp/src/`

   If you had previously opened Claude, then `File` > `Exit` and open it again.

## Usage

### Starting the Connection

1. Start your Jupyter Notebook (version 6.x) server:

   ```bash
   uv run jupyter nbclassic
   ```

2. Create a new Jupyter Notebook and make sure that you choose the `jupyter-mcp` kernel: `kernel` -> `change kernel` -> `jupyter-mcp`

3. In a notebook cell, run the following code to initialize the WebSocket server:

   ```python
   import sys
   sys.path.append('/path/to/jupyter-notebook-mcp/src')  # Add the path to where the scripts are located
   
   from jupyter_ws_server import setup_jupyter_mcp_integration
   
   # Start the WebSocket server inside Jupyter
   server, port = setup_jupyter_mcp_integration()
   ```

   Don't forget to replace here `'/path/to/jupyter-notebook-mcp/src'` with `src` folder on your system. For example:
   - Windows: `"C:\\Users\\MyUser\\GitHub\\jupyter-notebook-mcp\\src\\"`
   - Mac: `/Users/MyUser/GitHub/jupyter-notebook-mcp/src/`

   ![Notebook setup](/assets/img/notebook-setup.png)

4. Launch Claude desktop with MCP enabled.

### Using with Claude

Once connected, Claude will have access to the following tools:

- `ping` - Check server connectivity
- `insert_and_execute_cell` - Insert a cell at the specified position and execute it
- `save_notebook` - Save the current Jupyter notebook
- `get_cells_info` - Get information about all cells in the notebook
- `get_notebook_info` - Get information about the current notebook
- `run_cell` - Run a specific cell by its index
- `run_all_cells` - Run all cells in the notebook
- `get_cell_text_output` - Get the output content of a specific cell
- `get_image_output` - Get the images output of a specific cell
- `edit_cell_content` - Edit the content of an existing cell
- `set_slideshow_type`- Set the slide show type for cell

## ⚠️ DISCLAIMER

This is an experimental project and should be used with caution. This tool runs arbitrary Python code in your computer, which could potentially modify or delete data if not used carefully. Always back up your important projects and data.

## Example Prompts

Ask Claude to perform notebook operations:

### Python example

You can check the [example notebook](/notebooks/example_notebook.ipynb) and the [video demo](https://x.com/jjsantoso/status/1906780778807390562)

```plain
You have access to a Jupyter Notebook server.

I need to create a presentation about Python's Seaborn library.  
The content is as follows:

- What is Seaborn?
- Long vs. Wide data format
- Advantages of Seaborn over Matplotlib
- Commonly used Seaborn functions
- Live demonstration (comparison of Seaborn vs. Matplotlib)
  - Bar plot
  - Line plot
  - Scatter plot

For each concept, I want the main explanations provided in markdown cells, followed by one or more Python code cells demonstrating its usage. Keep the text concise—the cells shouldn't exceed 10 lines each.

Use appropriate slideshow types for each cell to make the presentation visually appealing.
```

[Check Here the full conversation](https://claude.ai/share/420b6aa6-b84b-437f-a6a6-89d310c36d52)

### Stata example

For this example, you need the [Stata Software](https://www.stata.com/) (v17 or later), which is not open source. If you already have Stata, you need to install the [`stata-setup`](https://pypi.org/project/stata-setup/) package:

```bash
uv pip install stata-setup
```

Then, at the begining of your notebook, you need to additionally include:

```python
import stata_setup
stata_setup.config('your_stata_installation_directory', 'your_stata_edition')
```

You can check the [example notebook](/notebooks/stata_example.ipynb) and the [video demo](https://x.com/jjsantoso/status/1906780784800731251)

This exercise comes from [Professor John Robert Warren webpage](https://www.rob-warren.com/soc3811_stata_exercises.html)

```plain
You have access to a Jupyter Notebook server. By default it runs Python, but you can run Stata (v18) code in this server using the %%stata magic, for example:

%%stata
display "hello world"

Run the available tools to solve the exercise, execute the code, and interpret the results.

**EXERCISE:**

In this exercise, you will use data from the American Community Survey (ACS). The ACS is a product of the U.S. Census Bureau and involves interviewing millions of Americans each year. For an introduction to the ACS, visit the ACS website (here).

For this exercise, I have created a data file containing two variables collected from respondents of the 2010 ACS who lived in one of two metropolitan areas: Minneapolis/St Paul and Duluth/Superior. The two variables are: (1) People's poverty status and (2) the time it takes people to commute to work.

Use STATA syntax files you already have (from the first assignment or class examples) and modify them to accomplish the following goals.

1. Read the data file (`"./stata_assignment_2.dat"`) for this assignment into STATA.
2. Be sure to declare "zero" as a missing value for `TRANTIME`, the commuting time variable.
3. Create a new dichotomous poverty variable that equals "1" if a person's income-to-poverty-line ratio (`POVRATIO`) is less than 100, and "0" otherwise; see the bottom of the assignment for an example of how to do this in STATA.
4. Separately for Minneapolis/St Paul and Duluth/Superior, produce:
   - a histogram of the commuting time (`TRANTIME`) variable.
   - measures of central tendency and spread for commuting time.
   - a frequency distribution for the poverty status (0 vs 1) variable.
5. Separately for Minneapolis/St Paul and Duluth/Superior, use STATA code to produce:
   - a 95% confidence interval for the mean commuting time.
   - a 95% confidence interval for the proportion of people who are poor. See below for an example of how to do this in STATA.

Use the results from step #4 above to:

6. Separately for Minneapolis/St Paul and Duluth/Superior, manually calculate:
   - a 95% confidence interval for the mean commuting time.
   - a 95% confidence interval for the proportion of people who are poor.
7. Confirm that your answers from steps #5 and #6 match.

Based on the results above, answer this question:

8. How do you interpret the confidence intervals calculated in steps #5 and #6 above?

9. Finally, create a do file (.do) with the all the Stata code and the answers as comments.

---

**DESCRIPTION OF VARIABLES IN "STATA ASSIGNMENT 2.DAT"**

**METAREAD** (Column 4-7)  
Metropolitan Area  
- `2240`: Duluth-Superior, MN/WI  
- `5120`: Minneapolis-St. Paul, MN  

**POVRATIO** (Column 18-20)  
Ratio of person's income to the poverty threshold:  
- `<100`: Below Poverty Line  
- `100`: At Poverty Line  
- `>100`: Above Poverty Line  

**TRANTIME** (Column 21-23)  
Travel time to work  
- `0`: Zero minutes  
- `1`: 1 Minute  
- etc.

```

[Check Here the full conversation](https://claude.ai/share/97b5a546-9375-434d-8224-561706782880)

## Testing with External Client

You can test the functionality without using Claude Desktop with the included external client:

```bash
uv run python src/jupyter_ws_external_client.py
```

This will provide an interactive menu to test some available functions.

For automated testing of all commands:

```bash
uv run python src/jupyter_ws_external_client.py --batch
```

## Troubleshooting

- **Connection Issues**: If you experience connection timeouts, the client includes a reconnection mechanism. You can also try restarting the WebSocket server.
- **Cell Execution Problems**: If cell execution doesn't work, check that the cell content is valid Python/Markdown and that the notebook kernel is running.
- **WebSocket Port Conflicts**: If the default port (8765) is already in use, the server will automatically try to find an available port.

## Limitations

- Only supports Jupyter Notebook 6.x
- Text output from cells is limited to 1500 characters by default
- Does not support advanced Jupyter widget interactions
- Connection may timeout after periods of inactivity

## License

[MIT](/LICENSE)

## Other Jupyter MCPs

This project is inspired by similar MCP integrations for Jupyter as:

- [ihrpr](https://github.com/ihrpr/mcp-server-jupyter)
- [Datalayer](https://github.com/datalayer/jupyter-mcp-server/tree/main)

# Grokking on Modular Addition in Mechanistic Interpretability

Training a 1-layer transformer on modular addition (a + b mod 113) and analyzing the grokking phenomenon, where the model first memorizes the training data, then suddenly generalizes to the test set after continued training.

Mutating code from Neel Nanda, TransformerLens, devinterp, and others. https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Grokking_Demo.ipynb , https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/grokking.ipynb , https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Grokking_Analysis.ipynb 
 
## Environment

**Always activate the local venv before running any commands:**

```bash
source .venv/bin/activate
```

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

## Paper: Progress Measures for Grokking via Mechanistic Interpretability

The file `paper.pdf` contains the primary reference paper (Nanda et al., ICLR 2023). Key details:

- **Authors**: Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
- **Published**: ICLR 2023 (arXiv:2301.05217)
- **Core contribution**: Fully reverse-engineers the algorithm learned by small transformers on modular addition, defines continuous progress measures that reveal grokking as a gradual process rather than a sudden shift

### Paper's Key Findings

- **Fourier Multiplication Algorithm**: The learned algorithm maps inputs onto a circle using discrete Fourier transforms, then uses trigonometric identities to convert addition to rotation. The model computes `cos(wk(a+b-c))` for key frequencies `wk = 2kπ/P`, which constructively interferes at the correct answer `c* = a+b mod P`
- **Key frequencies** (mainline model with P=113): `k ∈ {14, 35, 41, 42, 52}`. The embedding matrix `WE` and neuron-logit map `WL = WU·Wout` are sparse in the Fourier basis, concentrated on these frequencies
- **Neuron-logit map `WL`**: Approximately rank 10, with each direction corresponding to sin/cos of one of 5 key frequencies. Projecting MLP activations onto these directions yields `cos(wk(a+b))` and `sin(wk(a+b))` with >90% variance explained
- **Four lines of evidence**: (1) periodic structure in weights/activations, (2) `WL` decomposition matches trig identities, (3) individual neurons are degree-2 polynomials of sin/cos at a single frequency, (4) ablating key frequencies reduces performance to chance while ablating other 95% of frequencies slightly improves it
- **Three training phases**: Memorization → Circuit formation → Cleanup. The sudden test accuracy jump occurs during cleanup, *after* the generalizing mechanism is already learned. Progress measures (restricted loss, excluded loss) vary smoothly throughout
- **Robustness**: The Fourier multiplication algorithm is consistently learned across different random seeds, model depths (1-2 layers), training data fractions (10-90%), and prime moduli (P=53, 109, 113). Different seeds use different key frequencies but the same algorithmic structure
- **Regularization**: Weight decay is crucial for grokking. Without it, the model memorizes but never generalizes. Dropout also works; L1 regularization does not
- **Attention mechanism**: Each head's attention pattern is a lookup table on input token `a`, expressible as `σ(Cj[a] - Cj[b])` where `Cj` is sparse in the Fourier basis. The OV circuits transfer Fourier components to the residual stream

### Paper's Notation

- `P` = prime modulus (113 in mainline)
- `WE` = embedding matrix (d × P)
- `WU` = unembedding matrix (P × d)
- `Wout` = MLP output matrix (d × n)
- `WL = WU·Wout` = neuron-logit map (P × n)
- `wk = 2kπ/P` = frequency for index k
- `MLP(a,b)` = MLP activations for inputs a, b
- `Logits(a,b) ≈ WL · MLP(a,b)` (skip connection is negligible)

## References

- [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217) (Nanda et al., ICLR 2023) — `paper.pdf`
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) (Power et al., 2022)
