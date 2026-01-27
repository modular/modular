:description: The MAX Paged Cache API reference.
:title: paged_kv_cache
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/kv_cache/paged_kv_cache/

paged_kv_cache
--------------

Paged attention KV cache implementation with support for distributed inference.

This package provides the core implementation of paged KV cache management,
including cache managers, transfer engines for distributed settings, and tensor
parallelism support.

Modules
=======

* :code_link:`/max/api/python/kv_cache/paged_kv_cache/cache_manager|cache_manager`: Core paged KV cache manager implementation.
* :code_link:`/max/api/python/kv_cache/paged_kv_cache/tp_cache_manager|tp_cache_manager`: Tensor parallelism cache manager and input symbols.
* :code_link:`/max/api/python/kv_cache/paged_kv_cache/transfer_engine|transfer_engine`: KV cache transfer engine for distributed inference.

Classes
=======

* :code_link:`/max/api/python/kv_cache/paged_kv_cache/cache_manager|PagedKVCacheManager`: Manager for paged KV cache with data and tensor parallelism support.
* :code_link:`/max/api/python/kv_cache/paged_kv_cache/transfer_engine|KVTransferEngine`: Manages KV cache transfers between devices.
* :code_link:`/max/api/python/kv_cache/paged_kv_cache/transfer_engine|KVTransferEngineMetadata`: Metadata for transfer engine configuration.
* :code_link:`/max/api/python/kv_cache/paged_kv_cache/transfer_engine|TransferReqData`: Transfer request data structure.

.. toctree::
   :hidden:

   cache_manager
   tp_cache_manager
   transfer_engine
