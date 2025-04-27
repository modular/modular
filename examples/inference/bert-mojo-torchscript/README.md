# TorchScript BERT inference with Mojo

This directory includes scripts used to run simple BERT inference via the [MAX
Engine Mojo API](https://docs.modular.com/max/api/mojo/engine/) to
predict the masked words in a sentence.

## Quickstart

### Magic instructions

If you have [`magic`](https://docs.modular.com/magic), you can run the
following command:

```sh
magic run bash run.sh
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
```
