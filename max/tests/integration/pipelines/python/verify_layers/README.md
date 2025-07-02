# Layer-by-Layer Verification Tool

This tool performs layer-by-layer verification between MAX Engine and PyTorch
models to ensure numerical consistency and detect implementation differences.

## What It Does

The tool automatically runs **two verification scenarios** to provide
comprehensive analysis:

1. **Standard Mode (without input injection)**: Compares layer outputs when
   each model processes the same input independently
2. **Input Injection Mode**: Injects PyTorch layer inputs into corresponding
   MAX layers to isolate differences in layer implementations vs. accumulated
   errors

### Key Features

- **Execution Order Validation**: Verifies that MAX and PyTorch models execute
  layers in identical sequence
- **Numerical Metrics Focus**: Reports raw MSE values and statistics
- **Comprehensive Analysis**: Provides per-component statistics and
  layer-by-layer breakdowns

## How It Works

### Layer Capture Process

1. **MAX Model**: Monkey patches MAX modules to capture intermediate layer
   outputs during inference
2. **PyTorch Model**: Registers forward hooks on PyTorch modules to capture
   layer outputs
3. **Tensor Storage**: Saves all intermediate tensors with metadata (shapes,
   dtypes, execution order)

### Verification Process

1. **Layer Matching**: Maps corresponding layers between MAX and PyTorch using
   name normalization
2. **Execution Order Check**: Validates that both models execute layers in
   identical sequence
3. **Tensor Comparison**: Calculates MSE between corresponding layer outputs
   for numerical analysis
4. **Input Injection**: For the second scenario, loads PyTorch tensors and
   injects them into MAX layers

### Output Organization

if --save-layers enabled

```text
export_path/
├── without_injection/          # Standard verification scenario
│   ├── max_layers/             # MAX model outputs and metadata
│   ├── torch_layers/           # PyTorch model outputs and metadata
│   ├── module_names_max.txt    # module names for MAX implementation
│   ├── module_names_max.txt    # module names for PyTorch implementation
│   ├── mse_plot.png            # MSE visualization by component
│   └── layer_verification_report_*.txt  # Detailed numerical report
└── with_injection/             # Input injection scenario
│   ├── max_layers/             # MAX model outputs and metadata
│   ├── torch_layers/           # PyTorch model outputs and metadata
│   ├── module_names_max.txt    # module names for MAX implementation
│   ├── module_names_max.txt    # module names for PyTorch implementation
│   ├── mse_plot.png            # MSE visualization by component
│   └── layer_verification_report_*.txt  # Detailed numerical report
```

## Usage

```bash
# Basic usage - runs both scenarios automatically on gpu
br //SDK/integration-test/pipelines/python:verify_layers_e2e -- \
  --pipeline qwen3-32b --encoding bfloat16 --devices gpu 

# Custom output directory and save layer dumps
br //SDK/integration-test/pipelines/python:verify_layers_e2e -- \
  --pipeline qwen3-32b --encoding bfloat16 --devices gpu --save-layers --export-path /my/custom/output
```

## Understanding the Results

### Numerical Report Structure

The tool generates text reports focused purely on numerical metrics:

```text
EXECUTION ORDER VERIFICATION:
✅ Execution order check PASSED: 131 layers in identical order

SUMMARY:
  Total Layers: 131

MSE STATISTICS:
  Min: 1.234567e-12
  Max: 5.678901e-06
  Mean: 2.345678e-08
  Median: 1.234567e-09

PER-COMPONENT STATISTICS:
  attention:
    Layers: 64
    MSE Min: 1.234567e-12
    MSE Max: 2.345678e-08
    MSE Mean: 5.678901e-10

DETAILED LAYER RESULTS (Execution Order):
[  0] layers.0.input_layernorm
      MSE: 1.234567e-12
      Input shapes: [[1, 9, 5120]]
      Output shapes: [[1, 9, 5120]]
```

### Interpreting Results

#### MSE Visualization Plot

The tool generates `mse_plot.png` - a matplotlib visualization that provides
visual analysis of MSE values across different model components:

**Plot Features:**

- **X-axis**: Layer execution order (chronological sequence)
- **Y-axis**: MSE values (logarithmic scale for better visibility of small
  differences)
- **Color coding**: Different colors represent different component types
  (attention, MLP, normalization, etc.)
- **Scatter points**: Each point represents one layer's MSE value

**Interpretation Patterns:**

- **Outliers**: Layers with significantly higher MSE that may need
  investigation
- **Component grouping**: Visual separation between different layer types
  helps identify systematic differences
- **Trend analysis**: Whether MSE increases/decreases through the model depth

The plot is particularly useful for quickly identifying which layers or
components have the largest numerical differences and whether there are
systematic patterns in the discrepancies.

**Comparison Between Scenarios:**

- **Similar MSE in both scenarios**: Indicates consistent layer implementations
- **Much lower MSE with input injection**: Suggests accumulated error
  propagation in standard mode

## Supported Pipelines

Most of the pipelines included in SDK/integration-test/pipelines/python/generate_llm_logits.py

## Technical Details

### Input Injection Implementation

- Handles BFloat16 tensors by preserving dtype semantics while working around
  numpy limitations
- Converts PyTorch tensors to MAX TensorValues with proper device placement
- Maintains execution order and layer matching consistency

### Layer Name Normalization

- Maps naming conventions between MAX and PyTorch
- Handles hierarchical layer structures (e.g., `layers.0.attention` ↔
  `model.layers.0.attention`)
- Provides fallback naming for dynamically generated layers

## Expanding to New Models

To add support for a new model:

- **Add Special Rules**: If needed, add tensor processing rules for
   model-specific layers
