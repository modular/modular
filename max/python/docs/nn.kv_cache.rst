:title: max.nn.kv_cache
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn.kv\_cache
================

.. automodule:: max.nn.kv_cache
   :no-members:

.. currentmodule:: max.nn.kv_cache

Cache configuration
-------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVCacheBuffer
   KVCacheParamInterface
   KVCacheParams
   KVCacheQuantizationConfig
   MultiKVCacheParams

Cache inputs
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AttentionDispatchMetadata
   KVCacheInputs
   KVCacheInputsPerDevice
   NestedIterableDataclass
   PagedCacheValues

Attention dispatch
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AttentionDispatchMetadataScalars
   AttentionDispatchResolver

Metrics
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVCacheMetrics

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   attention_dispatch_metadata
   attention_dispatch_metadata_list
   build_max_lengths_tensor
   compute_max_seq_len_fitting_in_cache
   compute_num_device_blocks
   compute_num_host_blocks
   estimated_memory_size
   unflatten_ragged_attention_inputs
