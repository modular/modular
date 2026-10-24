:title: max.nn
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn
======

.. automodule:: max.nn
   :no-members:

.. currentmodule:: max.nn

Submodules
----------

.. autosummary::
   :nosignatures:

   max.nn.attention
   max.nn.comm
   max.nn.graph_groups
   max.nn.kernels
   max.nn.kv_cache
   max.nn.moe
   max.nn.sampling
   max.nn.state_space

.. toctree::
   :maxdepth: 1
   :hidden:

   nn.attention
   nn.comm
   nn.graph_groups
   nn.kernels
   nn.kv_cache
   nn.moe
   nn.sampling
   nn.state_space

Base classes
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Identity
   Layer
   LayerList
   MissingLayer
   Module
   Sequential
   Shardable
   Signals

Linear layers
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ColumnParallelLinear
   Embedding
   GPTQLinear
   Linear
   MLP
   StackedLinear
   VocabParallelEmbedding

Normalization
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ConstantLayerNorm
   GroupNorm
   LayerNorm
   RMSNorm

Rotary embeddings
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DynamicRotaryEmbedding
   LinearScalingParams
   Llama3RopeScalingParams
   Llama3RotaryEmbedding
   LongRoPERotaryEmbedding
   LongRoPEScalingParams
   RotaryEmbedding
   YarnRotaryEmbedding
   YarnScalingParams

Transformer
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DistributedTransformer
   DistributedTransformerBlock
   HyperConnection
   ReturnHiddenStates
   ReturnLogits
   Transformer
   TransformerBlock
   ~max.nn.transformer.DistributedLogitsPostprocessMixin
   ~max.nn.transformer.LogitsPostprocessMixin

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.nn.transformer.distributed_logits_postprocess
   ~max.nn.transformer.forward_sequential_layers
   ~max.nn.transformer.logits_outputs
   ~max.nn.transformer.logits_postprocess

Convolution
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Conv1D
   Conv2d
   Conv3D
   ConvTranspose1d
   WeightNormConvTranspose1d

Mixture of experts
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Allreduce
   ~max.nn.comm.ep.EPCommBuffers
   MoE
   MoEGate
   MoEQuantized

Parallelism
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.nn.parallel.ParallelArrayOps

Sampling
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   MinPSampler
   RejectionSampler
   RejectionSamplerWithResiduals

Quantization
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   QuantConfig
   QuantFormat
   WeightSource
   InputScaleSpec
   ScaleGranularity
   ScaleOrigin
   WeightScaleSpec

Hooks
-----

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.nn.hooks.PrintHook

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.nn.layer.add_layer_hook
   ~max.nn.layer.clear_hooks

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_max_lengths_tensors
   clamp
   forward_moe_sharded_layers
   split_batch
   split_batch_replicated
