# MAX Graph API examples

These examples demonstrate the flexibility of the
[MAX Graph API](https://docs.modular.com/max/graph/), a
[Mojo](https://docs.modular.com/mojo/) interface to the advanced graph compiler
within MAX.

## Magic instructions

If you have [`magic`](https://docs.modular.com/magic), you can run any of the
following commands:

```sh
magic run basic
```

## Conda instructions

```sh
# Create a Conda environment if you don't have one
conda create -n max-repo
# Update the environment with the environment.yml file
conda env update -n max-repo -f environment.yml --prune
# Run the example
conda activate max-repo

mojo basics/basic.🔥

conda deactivate
```

## [Graph API introduction](basics/)

A basic Mojo Graph API example that provides an introduction to how to
stage and run a computational graph on MAX, following the
[getting started guide](https://docs.modular.com/max/tutorials/get-started-with-max-graph).
