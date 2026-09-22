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

   CacheLeafParamInterface
   KVCacheBuffer
   KVCacheParamInterface
   KVCacheParams
   MHAKVCacheParams
   MLAKVCacheParams
   MSAKVCacheParams
   KVCacheQuantizationConfig
   KVConnectorType
   KVCacheMemory
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

   PagedKVLeafRegion
   RecurrentKVLeafRegion
   RecurrentLeafInputs
   RecurrentStateBuffer
   RecurrentStateInputsPerDevice
   RecurrentStateParams
   RecurrentStateRegion

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

   PACKED_PAGE_STRIDE

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
   padded_lut_cols
   recurrent_leaf
   spec_decode_cache_slack
