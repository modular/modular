# Docs code coverage

Generated at 2026-10-28 00:30 UTC.

## Summary

- 69/177 (39.0%) code blocks tested
- 11 pages with no tests

## Pages with no tests

- `develop/debugging.mdx`
- `develop/model-bringup-workflow.mdx`
- `develop/modules.mdx`
- `develop/pipelines.mdx`
- `develop/quantize.mdx`
- `develop/serve-custom-model-architectures.mdx`
- `develop/weight-adapters.mdx`
- `serve/chat-templates.mdx`
- `serve/function-calling.mdx`
- `serve/image-generation.mdx`
- `serve/video-generation.mdx`

## Pages with some tests

- `develop/build-custom-ops.mdx` (8/11)
- `develop/custom-kernels-pytorch.mdx` (0/4) (includes partial tests)
- `develop/custom-ops-matmul.mdx` (3/8)
- `develop/eager-execution.mdx` (3/5)
- `develop/graph.mdx` (0/2) (includes partial tests)
- `develop/layer-comparison.mdx` (0/3) (includes partial tests)
- `develop/layers.mdx` (3/5)
- `develop/logit-comparison.mdx` (0/1) (includes partial tests)
- `develop/read-model-config.mdx` (0/10) (includes partial tests)
- `gpu/block-and-warp.mdx` (0/4) (includes partial tests)
- `gpu/fundamentals.mdx` (3/9)
- `gpu/hardware-compatibility.mdx` (0/1) (includes partial tests)
- `gpu/intro-tutorial.mdx` (5/10)
- `layout/layouts.mdx` (1/4)
- `tile-tensor/layouts.mdx` (6/8)
- `tile-tensor/tensors.mdx` (2/19)

## Pages with full coverage

- `develop/basic-ops.mdx` (12/12)
- `develop/broadcasting.mdx` (4/4)
- `develop/dtypes.mdx` (9/9)
- `develop/indexing.mdx` (4/4)
- `develop/tensor-realization.mdx` (1/1)
- `develop/tensors.mdx` (5/5)

## Block details

### `develop/basic-ops.mdx` (12/12)

- L57 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/arithmetic.py`
- L85 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_math_ops.py`
- L120 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_reshape.py`
- L151 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_transpose.py`
- L181 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_transpose_t.py`
- L209 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_permute.py`
- L239 python: covered (100%) — `max/docs/site/code/develop/basic-ops/concat.py`
- L283 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_reductions.py`
- L328 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/tensor_matmul.py`
- L360 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/functional_activations.py`
- L396 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/random_basic.py`
- L420 python: covered (100%) —
  `max/docs/site/code/develop/basic-ops/compose_linear_layer.py`

### `develop/broadcasting.mdx` (4/4)

- L85 python: covered (100%) —
  `max/docs/site/code/develop/broadcasting/auto_broadcast.py`
- L111 python: covered (100%) —
  `max/docs/site/code/develop/broadcasting/mask_broadcast.py`
- L141 python: covered (100%) —
  `max/docs/site/code/develop/broadcasting/explicit_broadcast.py`
- L183 python: covered (100%) —
  `max/docs/site/code/develop/broadcasting/fix_rank.py`
- L220 python: skipped (3 or fewer lines)

### `develop/build-custom-ops.mdx` (8/11)

- L106 mojo: covered (100%) — `max/examples/custom_ops/kernels/add_one.mojo`
- L148 mojo: covered (100%) — `max/examples/custom_ops/kernels/add_one.mojo`
- L186 mojo: uncovered (20%)
- L216 python: uncovered (23.1%)
- L259 python: covered (100%) — `max/examples/custom_ops/addition.py`
- L305 python: covered (100%) — `max/examples/custom_ops/addition.py`
- L316 python: covered (100%) — `max/examples/custom_ops/addition.py`
- L355 mojo: covered (100%) — `max/examples/custom_ops/kernels/add_one.mojo`
- L377 python: covered (100%) —
  `max/examples/custom_ops/elementwise_addition.py`
- L431 mojo: covered (100%) —
  `max/examples/custom_ops/kernels/vector_addition.mojo`
- L460 mojo: partial (90%) —
  `max/examples/custom_ops/kernels/vector_addition.mojo`

### `develop/custom-kernels-pytorch.mdx` (0/4) (includes partial tests)

- L48 python: partial (83.3%) — `max/examples/pytorch_custom_ops/grayscale.py`
- L73 python: partial (80%) — `max/examples/pytorch_custom_ops/grayscale.py`
- L114 mojo: partial (93.3%) —
  `max/examples/pytorch_custom_ops/operations/grayscale.mojo`
- L205 python: uncovered (22.2%)

### `develop/custom-ops-matmul.mdx` (3/8)

- L155 python: covered (100%) —
  `max/examples/custom_ops/matrix_multiplication.py`
- L291 mojo: skipped (3 or fewer lines)
- L315 mojo: skipped (3 or fewer lines)
- L344 mojo: covered (100%) —
  `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- L387 mojo: skipped (3 or fewer lines)
- L433 mojo: skipped (3 or fewer lines)
- L459 mojo: uncovered (14.3%)
- L473 mojo: skipped (3 or fewer lines)
- L490 mojo: covered (100%) —
  `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- L519 mojo: skipped (3 or fewer lines)
- L558 mojo: partial (70%) —
  `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- L617 mojo: partial (76.9%) —
  `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- L685 mojo: uncovered (40%)
- L775 mojo: partial (88.2%) —
  `max/examples/custom_ops/kernels/matrix_multiplication.mojo`

### `develop/dtypes.mdx` (9/9)

- L17 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/dtype_intro.py`
- L56 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/specify_dtype.py`
- L90 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/check_dtype.py`
- L122 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/cast_dtype.py`
- L152 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/dtype_properties.py`
- L188 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/dlpack_conversion.py`
- L222 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/pytorch_to_max.py`
- L294 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/calculate_memory.py`
- L325 python: covered (100%) —
  `max/docs/site/code/develop/dtypes/validate_dtypes.py`

### `develop/eager-execution.mdx` (3/5)

- L26 python: covered (100%) —
  `max/docs/site/code/develop/index/eager_simple.py`
- L54 python: covered (100%) —
  `max/docs/site/code/develop/index/arithmetic_ops.py`
- L71 python: covered (100%) —
  `max/docs/site/code/develop/index/functional_api.py`
- L92 python: partial (92.9%) —
  `max/docs/site/code/develop/index/when_to_use_eager.py`
- L131 python: partial (91.7%) —
  `max/docs/site/code/develop/index/eager_workflow.py`

### `develop/graph.mdx` (0/2) (includes partial tests)

- L90 python: uncovered (40%)
- L117 python: partial (93.8%) —
  `max/docs/site/code/develop/index/graph_simple.py`

### `develop/indexing.mdx` (4/4)

- L41 python: covered (100%) — `max/docs/site/code/develop/indexing/slicing.py`
- L72 python: covered (100%) — `max/docs/site/code/develop/indexing/gather.py`
- L113 python: covered (100%) — `max/docs/site/code/develop/indexing/where.py`
- L157 python: covered (100%) — `max/docs/site/code/develop/indexing/scatter.py`

### `develop/layer-comparison.mdx` (0/3) (includes partial tests)

- L92 python: partial (80%) —
  `max/docs/site/code/develop/layer-comparison/layer_comparison.py`
- L176 python: partial (73.1%) —
  `max/docs/site/code/develop/layer-comparison/layer_comparison.py`
- L224 python: partial (92.5%) —
  `max/docs/site/code/develop/layer-comparison/layer_comparison.py`

### `develop/layers.mdx` (3/5)

- L39 python: covered (100%) —
  `max/docs/site/code/develop/layers/subclass_rms_norm.py`
- L82 python: covered (100%) —
  `max/docs/site/code/develop/layers/define_module.py`
- L135 python: covered (100%) —
  `max/docs/site/code/develop/layers/compose_and_run.py`
- L214 python: uncovered (25%)
- L236 python: uncovered (25%)

### `develop/logit-comparison.mdx` (0/1) (includes partial tests)

- L142 python: partial (91.1%) —
  `max/docs/site/code/develop/logit-comparison/logit_comparison.py`

### `develop/read-model-config.mdx` (0/10) (includes partial tests)

- L52 python: skipped (3 or fewer lines)
- L226 python: uncovered (15.4%)
- L307 python: uncovered (5.3%)
- L366 python: partial (61.5%) —
  `max/examples/custom-models/qwen2/model_config.py`
- L405 python: uncovered (0%)
- L422 python: uncovered (5%)
- L454 python: uncovered (28.6%)
- L526 python: uncovered (22.2%)
- L544 python: uncovered (10%)
- L567 python: skipped (3 or fewer lines)
- L577 python: uncovered (25%)
- L604 python: uncovered (5.9%)

### `develop/tensor-realization.mdx` (1/1)

- L71 python: covered (100%) —
  `max/docs/site/code/develop/tensor-realization/functional_layer.py`

### `develop/tensors.mdx` (5/5)

- L49 python: skipped (3 or fewer lines)
- L83 python: covered (100%) —
  `max/docs/site/code/develop/tensors/create_multi_dim.py`
- L110 python: covered (100%) —
  `max/docs/site/code/develop/tensors/create_factory.py`
- L138 python: skipped (3 or fewer lines)
- L157 python: skipped (3 or fewer lines)
- L189 python: covered (100%) —
  `max/docs/site/code/develop/tensors/properties_shape.py`
- L222 python: covered (100%) —
  `max/docs/site/code/develop/tensors/properties_rank.py`
- L241 python: covered (100%) —
  `max/docs/site/code/develop/tensors/properties_dtype.py`
- L260 python: skipped (3 or fewer lines)
- L280 python: skipped (3 or fewer lines)

### `gpu/block-and-warp.mdx` (0/4) (includes partial tests)

- L243 mojo: partial (92%) — `max/examples/gpu-block-and-warp/tiled_matmul.mojo`
- L833 mojo: skipped (3 or fewer lines)
- L846 mojo: skipped (3 or fewer lines)
- L887 mojo: uncovered (16.7%)
- L900 mojo: uncovered (16.7%)
- L959 mojo: uncovered (10%)

### `gpu/fundamentals.mdx` (3/9)

- L56 mojo: covered (100%) —
  `max/docs/site/code/gpu/fundamentals/detect_gpu.mojo`
- L115 mojo: partial (60.4%) —
  `max/docs/site/code/gpu/fundamentals/scalar_add.mojo`
- L290 mojo: covered (100%) —
  `max/docs/site/code/gpu/fundamentals/print_threads.mojo`
- L337 mojo: covered (100%) —
  `max/docs/site/code/gpu/fundamentals/print_threads.mojo`
- L468 mojo: skipped (3 or fewer lines)
- L512 mojo: skipped (3 or fewer lines)
- L557 mojo: skipped (3 or fewer lines)
- L585 mojo: uncovered (25%)
- L629 mojo: skipped (3 or fewer lines)
- L646 mojo: uncovered (16.7%)
- L662 mojo: uncovered (30%)
- L706 mojo: partial (90%) —
  `max/docs/site/code/gpu/fundamentals/scalar_add.mojo`
- L753 mojo: partial (85.7%) —
  `max/docs/site/code/gpu/fundamentals/scalar_add.mojo`

### `gpu/hardware-compatibility.mdx` (0/1) (includes partial tests)

- L332 mojo: partial (75%) —
  `max/docs/site/code/gpu/fundamentals/print_threads.mojo`

### `gpu/intro-tutorial.mdx` (5/10)

- L116 mojo: partial (87.5%) —
  `max/docs/site/code/gpu/intro-tutorial/detect_gpu.mojo`
- L193 mojo: partial (75%) —
  `max/docs/site/code/gpu/intro-tutorial/print_threads.mojo`
- L211 mojo: partial (82.4%) —
  `max/docs/site/code/gpu/intro-tutorial/print_threads.mojo`
- L385 mojo: covered (100%) —
  `max/docs/site/code/gpu/intro-tutorial/print_threads.mojo`
- L404 mojo: partial (71.4%) —
  `max/docs/site/code/gpu/intro-tutorial/print_threads.mojo`
- L483 mojo: partial (90.9%) — `max/examples/gpu-intro/vector_addition.mojo`
- L568 mojo: covered (100%) — `max/examples/gpu-intro/vector_addition.mojo`
- L620 mojo: skipped (3 or fewer lines)
- L642 mojo: skipped (3 or fewer lines)
- L652 mojo: skipped (3 or fewer lines)
- L659 mojo: skipped (3 or fewer lines)
- L671 mojo: skipped (3 or fewer lines)
- L679 mojo: covered (100%) — `max/examples/gpu-intro/vector_addition.mojo`
- L742 mojo: covered (100%) — `max/examples/gpu-intro/vector_addition.mojo`
- L771 mojo: covered (100%) — `max/examples/gpu-intro/vector_addition.mojo`

### `layout/layouts.mdx` (1/4)

- L45 mojo: skipped (3 or fewer lines)
- L76 mojo: skipped (3 or fewer lines)
- L103 mojo: partial (50%) — `max/examples/layouts/basic_layouts.mojo`
- L138 mojo: skipped (3 or fewer lines)
- L306 mojo: skipped (3 or fewer lines)
- L321 mojo: partial (60%) — `max/examples/layouts/tiled_layouts.mojo`
- L365 mojo: skipped (3 or fewer lines)
- L398 mojo: covered (100%) — `max/examples/layouts/tiled_layouts.mojo`
- L450 mojo: partial (60%) — `max/examples/layouts/tiled_layouts.mojo`

### `tile-tensor/layouts.mdx` (6/8)

- L39 mojo: partial (70%) —
  `max/docs/site/code/tile-tensor/layouts/tiled_layouts.mojo`
- L68 mojo: skipped (3 or fewer lines)
- L97 mojo: skipped (3 or fewer lines)
- L124 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/layouts/basic_layouts.mojo`
- L155 mojo: skipped (3 or fewer lines)
- L175 mojo: skipped (3 or fewer lines)
- L185 mojo: partial (60%) —
  `max/docs/site/code/tile-tensor/layouts/basic_layouts.mojo`
- L197 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/layouts/basic_layouts.mojo`
- L335 mojo: skipped (3 or fewer lines)
- L342 mojo: skipped (3 or fewer lines)
- L360 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/layouts/basic_layouts.mojo`
- L416 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/layouts/tiled_layouts.mojo`
- L484 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/layouts/tiled_layouts.mojo`
- L552 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/layouts/tiled_layouts.mojo`

### `tile-tensor/tensors.mdx` (2/19)

- L71 mojo: skipped (3 or fewer lines)
- L94 mojo: uncovered (0%)
- L136 mojo: uncovered (33.3%)
- L152 mojo: skipped (3 or fewer lines)
- L187 mojo: skipped (3 or fewer lines)
- L196 mojo: skipped (3 or fewer lines)
- L238 mojo: partial (85.7%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L259 mojo: partial (75%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L288 mojo: partial (90.9%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L355 mojo: partial (84.8%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_init_from_cpu.mojo`
- L427 mojo: uncovered (20%)
- L506 mojo: partial (81.8%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L555 mojo: partial (68.8%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L598 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L623 mojo: partial (94.7%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L660 mojo: covered (100%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L700 mojo: partial (88.9%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L762 mojo: skipped (3 or fewer lines)
- L823 mojo: uncovered (20%)
- L840 mojo: uncovered (20%)
- L868 mojo: uncovered (42.9%)
- L886 mojo: uncovered (42.9%)
- L925 mojo: partial (85.7%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_examples.mojo`
- L995 mojo: partial (64.1%) —
  `max/docs/site/code/tile-tensor/tensors/tile_tensor_tile_copier.mojo`

## Untested blocks

### `develop/debugging.mdx` (0/1)

- L82 python: uncovered (14.3%)

### `develop/model-bringup-workflow.mdx` (0/9)

- L141 python: uncovered (11.1%)
- L278 python: uncovered (20%)
- L310 python: uncovered (0%)
- L352 python: uncovered (25%)
- L447 python: uncovered (15%)
- L500 python: uncovered (20%)
- L554 python: skipped (3 or fewer lines)
- L589 python: uncovered (0%)
- L604 python: skipped (3 or fewer lines)
- L635 python: skipped (3 or fewer lines)
- L683 python: uncovered (0%)
- L731 python: uncovered (6.5%)
- L780 python: skipped (3 or fewer lines)

### `develop/modules.mdx` (0/5)

- L59 python: uncovered (36%)
- L119 python: uncovered (33.3%)
- L150 python: uncovered (27.3%)
- L211 python: uncovered (31.6%)
- L262 python: skipped (3 or fewer lines)
- L276 python: uncovered (15.4%)

### `develop/pipelines.mdx` (0/6)

- L83 python: uncovered (8.3%)
- L173 python: uncovered (24.2%)
- L295 python: uncovered (22.2%)
- L337 python: uncovered (10%)
- L431 python: uncovered (0%)
- L565 python: uncovered (25%)

### `develop/quantize.mdx` (0/1)

- L72 python: uncovered (16.7%)

### `develop/serve-custom-model-architectures.mdx` (0/1)

- L38 python: skipped (3 or fewer lines)
- L61 python: uncovered (6.7%)
- L191 python: skipped (live inference client)

### `develop/weight-adapters.mdx` (0/9)

- L100 python: uncovered (14.3%)
- L125 python: uncovered (16.7%)
- L152 python: uncovered (12.5%)
- L218 python: uncovered (16.7%)
- L289 python: uncovered (0%)
- L332 python: uncovered (0%)
- L361 python: uncovered (0%)
- L381 python: skipped (3 or fewer lines)
- L434 python: uncovered (7.5%)
- L512 python: uncovered (16.7%)
- L565 python: skipped (3 or fewer lines)

### `serve/chat-templates.mdx` (0/1)

- L195 python: skipped (live inference client)
- L261 python: uncovered (31.2%)

### `serve/function-calling.mdx` (0/1)

- L59 python: skipped (live inference client)
- L127 python: skipped (3 or fewer lines)
- L147 py: uncovered (0%)
- L267 python: skipped (live inference client)

### `serve/image-generation.mdx` (0/3)

- L38 python: uncovered (20%)
- L80 python: uncovered (8.3%)
- L148 python: uncovered (7.4%)
- L381 python: skipped (live inference client)
- L483 python: skipped (live inference client)

### `serve/video-generation.mdx` (0/1)

- L38 python: uncovered (11.8%)
- L96 python: skipped (3 or fewer lines)
- L211 python: skipped (live inference client)

## Warnings

- `max/examples/custom-models/qwen2/model_config.py`
  (`develop/read-model-config.mdx`): matched a block on this page but its # DOC:
  header points elsewhere
- `max/docs/site/code/gpu/fundamentals/print_threads.mojo`
  (`gpu/hardware-compatibility.mdx`): matched a block on this page but its #
  DOC: header points elsewhere
- `max/docs/site/code/develop/tensors/create_first_tensor.py`
  (`develop/tensors.mdx`): matches no Python or Mojo block on its page
- `max/docs/site/code/develop/tensors/create_random.py` (`develop/tensors.mdx`):
  matches no Python or Mojo block on its page
- `max/docs/site/code/develop/tensors/create_random_mean_std.py`
  (`develop/tensors.mdx`): matches no Python or Mojo block on its page
- `max/docs/site/code/develop/tensors/properties_device.py`
  (`develop/tensors.mdx`): matches no Python or Mojo block on its page
- `max/docs/site/code/develop/tensors/properties_num_elements.py`
  (`develop/tensors.mdx`): matches no Python or Mojo block on its page
- `max/docs/site/code/gpu/fundamentals/device_info.mojo`
  (`gpu/fundamentals.mdx`): matches no Python or Mojo block on its page
- `max/docs/site/code/tile-tensor/tensors/tile_tensor_gpu_examples.mojo`
  (`tile-tensor/tensors.mdx`): matches no Python or Mojo block on its page
- `max/examples/custom-models/qwen2/__init__.py`
  (`develop/serve-custom-model-architectures.mdx`): matches no Python or Mojo
  block on its page
- `max/examples/custom-models/qwen2/arch.py`
  (`develop/serve-custom-model-architectures.mdx`): matches no Python or Mojo
  block on its page
- `max/examples/custom-models/qwen2/model.py`
  (`develop/serve-custom-model-architectures.mdx`): matches no Python or Mojo
  block on its page
- `max/examples/custom-models/qwen2/model_config.py`
  (`develop/serve-custom-model-architectures.mdx`): matches no Python or Mojo
  block on its page
- `max/examples/modules/modules.py` (`develop/modules.mdx`): matches no Python
  or Mojo block on its page
- `max/examples/pytorch_custom_ops/torch-grayscale.py`
  (`develop/custom-kernels-pytorch.mdx`): matches no Python or Mojo block on its
  page
