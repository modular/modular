:description: The MAX Python Neural Network API reference.
:title: nn
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/nn/

MAX Neural Network API
----------------------

APIs to build neural network components for deep learning models with Python.

The MAX neural network API provides two namespaces:

- **max.nn**: Graph-based API for building computational graphs.
- **max.experimental.nn**: Eager-style execution with PyTorch-style syntax.

For functional operations like relu, softmax, and more, see the
:code_link:`/max/api/python/experimental/functional|functional` module.

Graph API
=========

Use these modules for building graph-based neural networks.

* :code_link:`/max/api/python/nn/attention|attention`: Attention mechanisms for sequence modeling.
* :code_link:`/max/api/python/nn/clamp|clamp`: Value clamping utilities for tensor operations.
* :code_link:`/max/api/python/nn/comm|comm`: Communication primitives for distributed training.
* :code_link:`/max/api/python/nn/conv|conv`: Convolutional layers for spatial processing.
* :code_link:`/max/api/python/nn/conv_transpose|conv_transpose`: Transposed convolution for upsampling.
* :code_link:`/max/api/python/nn/data_parallelism|data_parallelism`: Utilities for splitting batches across devices.
* :code_link:`/max/api/python/nn/embedding|embedding`: Embedding layers with vocabulary support.
* :code_link:`/max/api/python/nn/float8_config|float8_config`: Configuration for FP8 quantization.
* :code_link:`/max/api/python/nn/hooks|hooks`: Extension hooks for layer customization.
* :code_link:`/max/api/python/nn/identity|identity`: Identity layer that passes inputs through unchanged.
* :code_link:`/max/api/python/nn/kernels|kernels`: Custom kernel implementations.
* :code_link:`/max/api/python/nn/kv_cache|kv_cache`: Key-value cache for efficient generation.
* :code_link:`/max/api/python/nn/layer|layer`: Base classes for building graph-based layers.
* :code_link:`/max/api/python/nn/linear|linear`: Linear transformation layers with optional parallelism.
* :code_link:`/max/api/python/nn/lora|lora`: Low-Rank Adaptation for efficient fine-tuning.
* :code_link:`/max/api/python/nn/moe|moe`: Mixture of Experts layer implementations.
* :code_link:`/max/api/python/nn/norm|norm`: Normalization layers for training stability.
* :code_link:`/max/api/python/nn/rotary_embedding|rotary_embedding`: Rotary position embeddings for sequences.
* :code_link:`/max/api/python/nn/sampling|sampling`: Sampling strategies for generation.
* :code_link:`/max/api/python/nn/sequential|sequential`: Container for sequential layer composition.
* :code_link:`/max/api/python/nn/transformer|transformer`: Transformer building blocks and layers.

Eager API
=========

.. note::
   The eager API provides PyTorch-style execution. Import from
   ``max.experimental.nn`` — see
   :code_link:`/max/api/python/experimental/nn|max.experimental.nn` for
   the full reference.

   To load the eager architecture variant in the inference server, pass
   ``--prefer-module-v3`` to ``max serve`` or ``max generate``.


.. toctree::
   :hidden:

   clamp
   comm/index
   conv
   conv_transpose
   data_parallelism
   embedding
   float8_config
   hooks
   identity
   kernels
   kv_cache/index
   layer
   linear
   lora
   moe
   norm/index
   rotary_embedding
   sampling
   sequential
   transformer/index
   attention/index
