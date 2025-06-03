# Extending PyTorch with custom operations in Mojo

> [!NOTE]
> This is a preview of the capability to write PyTorch custom operations in
> Mojo, and we plan to enhance the performance and ergonomics of this feature.

Custom operations in PyTorch can now be written using Mojo, letting you
experiment with new GPU algorithms in a familiar PyTorch environment. These
custom operations are registered using the
[`CustomOpLibrary`](https://docs.modular.com/max/api/python/torch/#max.torch.CustomOpLibrary)
class in the `max.torch` package.

These examples show how to register and use Mojo custom operations in PyTorch
models, from very basic calculations to complex image processing and full model
implementations. These examples require a system with a [MAX-compatible
GPU](https://docs.modular.com/max/faq/#gpu-requirements)

The three examples of PyTorch custom operations in Mojo consist of:

- **addition.py**: A very basic example, where a custom operation that adds a
  constant value to every element of an input tensor is defined and run.
- **grayscale.py**: An image processing example that converts RGB images to
  grayscale using a custom Mojo kernel. This demonstrates how to process
  real-world data with automatic vectorization and parallelization.
- **whisper.py**: A PyTorch implementation of the
  [Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper)
  speech model, with its attention layer replaced by a custom fused attention
  operation defined in Mojo. This shows how Mojo custom operations could be
  integrated into the context of a larger PyTorch model.

You can run these examples via [Pixi](https://pixi.sh):

```sh
pixi run addition
pixi run grayscale
pixi run whisper
```

or directly in a Python virtual environment where the `max` PyPI package and
PyTorch have been installed:

```sh
python addition.py
python grayscale.py
python whisper.py
```
