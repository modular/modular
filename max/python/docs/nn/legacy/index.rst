:description: The MAX Python legacy neural network API reference.
:title: legacy
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/nn/legacy/

max.nn.legacy
-------------

Legacy graph-based neural network API.

.. note::
   This is the legacy API for backward compatibility. For all new models, use
   the eager tensor API from [nn](/max/api/python/nn).

The legacy API provides graph-based layer implementations for building neural
networks. This API was the primary interface prior to MAX 26.1 and remains
available for backward compatibility.

**Using the Legacy API:**

.. code-block:: python

   from max.nn.legacy import Module, Linear, LayerNorm
   from max.nn.legacy.attention import AttentionWithRope

Modules
=======

* :code_link:`/max/api/python/nn/legacy/attention|attention`: Attention mechanisms for sequence modeling.
* :code_link:`/max/api/python/nn/legacy/clamp|clamp`: Value clamping utilities for tensor operations.
* :code_link:`/max/api/python/nn/legacy/comm|comm`: Communication primitives for distributed training.
* :code_link:`/max/api/python/nn/legacy/conv|conv`: Convolutional layers for spatial processing.
* :code_link:`/max/api/python/nn/legacy/conv_transpose|conv_transpose`: Transposed convolution for upsampling.
* :code_link:`/max/api/python/nn/legacy/embedding|embedding`: Embedding layers with vocabulary support.
* :code_link:`/max/api/python/nn/legacy/float8_config|float8_config`: Configuration for FP8 quantization.
* :code_link:`/max/api/python/nn/legacy/hooks|hooks`: Extension hooks for layer customization.
* :code_link:`/max/api/python/nn/legacy/kernels|kernels`: Custom kernel implementations.
* :code_link:`/max/api/python/nn/legacy/kv_cache|kv_cache`: Key-value cache for efficient generation.
* :code_link:`/max/api/python/nn/legacy/layer|layer`: Base classes for building graph-based layers.
* :code_link:`/max/api/python/nn/legacy/linear|linear`: Linear transformation layers with optional parallelism.
* :code_link:`/max/api/python/nn/legacy/lora|lora`: Low-Rank Adaptation for efficient fine-tuning.
* :code_link:`/max/api/python/nn/legacy/moe|moe`: Mixture of Experts layer implementations.
* :code_link:`/max/api/python/nn/legacy/norm|norm`: Normalization layers for training stability.
* :code_link:`/max/api/python/nn/legacy/rotary_embedding|rotary_embedding`: Rotary position embeddings for sequences.
* :code_link:`/max/api/python/nn/legacy/sampling|sampling`: Sampling strategies for generation.
* :code_link:`/max/api/python/nn/legacy/sequential|sequential`: Container for sequential layer composition.
* :code_link:`/max/api/python/nn/legacy/transformer|transformer`: Transformer building blocks and layers.

.. toctree::
   :hidden:

   clamp
   comm/index
   conv
   conv_transpose
   embedding
   float8_config
   hooks
   kernels
   kv_cache/index
   layer
   linear
   lora
   moe
   norm
   rotary_embedding
   sampling
   sequential
   transformer/index
   attention/index
