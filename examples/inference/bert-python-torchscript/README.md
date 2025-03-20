# TorchScript BERT inference with Python

This directory includes scripts used to run simple BERT inference via the MAX
Engine Python API to predict the masked words in a sentence.

## Quickstart

### Magic instructions

If you have [`magic`](https://docs.modular.com/magic), you can run the
following command:

```sh
# Run the MAX Engine example
magic run bash run.sh
# Run the MAX Serving example
magic run bash deploy.sh
```

### Conda instructions

Create a Conda environment, activate that environment, and install the
requirements:

```sh
# Create a Conda environment if you don't have one
conda create -n max-repo
# Update the environment with the environment.yml file
conda env update -n max-repo -f environment.yml --prune
# Run the example
conda run -n max-repo --live-stream bash run.sh
# Run the MAX Serving example
conda run -n max-repo --live-stream bash deploy.sh
```

## Scripts included

- `simple-inference.py`: Predicts the masked words in the input text using the
MAX Engine Python API. The script prepares an example input, executes the
model, and generates the filled mask.

    You can use the `--text` CLI flag to specify an input sentence.
    For example:

    ```sh
    python3 simple-inference.py --text "Paris is the [MASK] of France."
    ```
