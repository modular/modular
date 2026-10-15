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

.. autosummary::
   :nosignatures:

   max.pipelines.kv_cache.paged_kv_cache

.. toctree::
   :maxdepth: 1
   :hidden:

   pipelines.kv_cache.paged_kv_cache

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

   BlockCount
   ByteCount
   DummyKVCache
   InsufficientBlocksError
   JengaKVCacheManager
   PagedKVCacheManager
   PagedKVCacheManagerInterface

Connectors
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.pipelines.kv_cache.connectors.KVConnector
   ~max.pipelines.kv_cache.connectors.NullConnector
   ~max.pipelines.kv_cache.connectors.dkv.DKVConnector

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.pipelines.kv_cache.connectors.create_connector

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
   kv_cache_memory_size
   load_kv_manager
   max_seq_len_fitting_in_cache

Utilities
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   cache_dtype_for_encoding
   is_fp4_kv_cache_format
