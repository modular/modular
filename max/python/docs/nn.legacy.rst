:title: max.nn.legacy
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn.legacy
=============

.. automodule:: max.nn.legacy
   :no-members:

.. currentmodule:: max.nn.legacy

Submodules
----------

.. autosummary::
   :nosignatures:

   max.nn.legacy.attention
   max.nn.legacy.comm
   max.nn.legacy.graph_groups
   max.nn.legacy.kernels
   max.nn.legacy.kv_cache
   max.nn.legacy.moe
   max.nn.legacy.sampling
   max.nn.legacy.state_space

.. toctree::
   :maxdepth: 1
   :hidden:

   nn.legacy.attention
   nn.legacy.comm
   nn.legacy.graph_groups
   nn.legacy.kernels
   nn.legacy.kv_cache
   nn.legacy.moe
   nn.legacy.sampling
   nn.legacy.state_space

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
   ~max.nn.legacy.transformer.DistributedLogitsPostprocessMixin
   ~max.nn.legacy.transformer.LogitsPostprocessMixin

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.nn.legacy.transformer.distributed_logits_postprocess
   ~max.nn.legacy.transformer.forward_sequential_layers
   ~max.nn.legacy.transformer.logits_outputs
   ~max.nn.legacy.transformer.logits_postprocess

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
   ~max.nn.legacy.comm.ep.EPCommBuffers
   MoE
   MoEGate
   MoEQuantized

Parallelism
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.nn.legacy.parallel.ParallelArrayOps

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

   ~max.nn.legacy.hooks.PrintHook

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.nn.legacy.layer.add_layer_hook
   ~max.nn.legacy.layer.clear_hooks

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
