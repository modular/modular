:title: max.pipelines.kv_cache
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.kv_cache
======================

.. automodule:: max.pipelines.kv_cache
   :no-members:

.. currentmodule:: max.pipelines.kv_cache

Submodules
----------

.. toctree::
   :maxdepth: 1

   pipelines.kv_cache.paged_kv_cache.kv_group_coordinator
   pipelines.kv_cache.paged_kv_cache.recurrent_coordinator

Memory planning
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   MemoryPlanner
   ModelConfig
   ModelConfigWithKVCache
   PagedMemoryPlanner

Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVCacheConfig
   KVConnectorConfig

Cache manager
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ByteCount
   DummyKVCache
   InsufficientBlocksError
   PagedKVCacheManager

Transfer engine
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVTransferEngine
   KVTransferEngineMetadata
   TransferReqData

Factory functions
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   available_port
   load_kv_manager
   max_seq_len_fitting_in_cache

Utilities
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   cache_dtype_for_encoding
