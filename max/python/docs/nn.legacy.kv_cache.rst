:title: max.nn.legacy.kv_cache
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn.legacy.kv\_cache
=======================

.. automodule:: max.nn.legacy.kv_cache
   :no-members:

.. currentmodule:: max.nn.legacy.kv_cache

Cache configuration
-------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   CacheLeafParamInterface
   HostStagingAlloc
   KVCacheAssignments
   KVCacheBuffer
   KVCacheBufferInterface
   KVCacheParamInterface
   KVCacheParams
   KVCacheGroupId
   MHAKVCacheParams
   MLAKVCacheParams
   MSAKVCacheParams
   KVCacheQuantizationConfig
   KVConnectorType
   KVConnectorConfigInterface
   NullKVConnectorConfig
   KVCacheMemory
   MultiKVCacheBuffer
   MultiKVCacheParams

Cache inputs
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVCacheInputs
   KVCacheInputsPerDevice
   BatchCharacteristics
   PagedCacheValues

Recurrent state
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   CacheLeafKind
   KVLeafRegion
   PagedKVLeafRegion
   RecurrentKVLeafRegion
   RecurrentLeafInputs
   RecurrentStateBuffer
   RecurrentStateInputsPerDevice
   RecurrentStateParams
   RecurrentStateRegion
   ScratchRecurrentKVLeafRegion

Attention dispatch
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AttnKey
   AttnKeyInterface
   MHAAttnKey
   MLAAttnKey
   MSAAttnKey

Metrics
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVCacheMetrics

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   KVHashAlgo

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_max_lengths_tensors
   compute_max_seq_len_fitting_in_cache
   compute_num_device_blocks
   estimated_memory_size
   max_pages_per_request
   packed_page_stride
   padded_lut_cols
   recurrent_leaf
   recurrent_leaves
   spec_decode_cache_slack
