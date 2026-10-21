# LayoutTensor examples

This directory contains examples to accompany
[Using LayoutTensor](https://max.modular.com/layout/tensors/).

`LayoutTensor` is deprecated as of MAX 26.7, so these examples compile with
deprecation warnings. For new code, use `TileTensor`, described in
[Using TileTensor](https://max.modular.com/tile-tensor/tensors/).

If you have [Pixi](https://pixi.sh/latest/) installed, you can
execute the examples by running the following commands:

```bash
pixi run layout_tensor
pixi run layout_tensor_gpu
pixi run layout_tensor_all  # Runs both sets of examples
```

Note that the GPU examples currently **don't** run on Apple silicon GPUs.

If you update any of these examples, note that the corresponding examples and
accompanying text in the MAX documentation needs to be updated as well.
