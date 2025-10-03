# Build custom neural network modules with MAX Python API

The [MAX Python API](https://docs.modular.com/max/api/python/) provides a
PyTorch-like interface for building neural network components that compile to
highly optimized graphs. This example shows how to create a reusable,
modular component using MAX's `nn.Module` class.

This example include custom layers, blocks, and architectural patterns that
showcase the flexibility of MAX's Python API for deep learning development, from
simple MLP blocks to more complex neural network architectures.

For a walkthrough, see the tutorial to [build an MLP block as a
module](https://docs.modular.com/max/tutorials/build-an-mlp-block/).

## Usage

If you don't have it, install `pixi`:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Then navigate to this directory and run the example:

```sh
pixi run python main.py
```

You should see the following output:

```output
--- Simple MLP Block ---
MLPBlock(1 linear layers, 1 activations)
------------------------------
--- MLP Block (1 Hidden Layer) ---
MLPBlock(2 linear layers, 1 activations)
------------------------------
--- Deeper MLP Block (3 Hidden Layers, GELU) ---
MLPBlock(4 linear layers, 3 activations)
------------------------------
```
