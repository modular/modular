# Basic example for MAX Graph API in Python

The [MAX graph API](https://docs.modular.com/max/graph/) provides a powerful
framework for staging computational graphs to be run on GPUs, CPUs, and more.
This is a basic example of building a model with the MAX Graph API purely in
Python and then executing it with MAX.

For more complex examples that use Mojo to write graph ops for GPUs and CPUs,
see the [custom CPU or GPU graph op examples](../custom_ops/).

For more explanation about this code, see the tutorial to [Get started with MAX
Graph in
Python](https://docs.modular.com/max/tutorials/get-started-with-max-graph-in-python).

## Usage

If you don't have it, install `pixi`:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Then navigate to this directory and run the example:

```sh
pixi run addition
```

You should see the following output:

```output
input names are:
name: input0, shape: [1], dtype: DType.float32
name: input1, shape: [1], dtype: DType.float32
result: [2.]
```
