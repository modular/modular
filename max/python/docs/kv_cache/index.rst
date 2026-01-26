:description: The MAX KV Cache API reference.
:title: kv_cache
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/kv_cache/

kv_cache
--------

KV cache management for efficient attention computation during inference.

This package provides implementations for managing key-value caches used in
transformer models. The paged attention implementation enables efficient memory
management by fragmenting cache memory into pages, allowing for better memory
utilization and support for prefix caching.

Functions
=========

* :code_link:`/max/api/python/kv_cache/registry|load_kv_manager`: Load and initialize a KV cache manager.
* :code_link:`/max/api/python/kv_cache/registry|estimate_kv_cache_size`: Estimate KV cache memory requirements.
* :code_link:`/max/api/python/kv_cache/registry|infer_optimal_batch_size`: Infer optimal batch size based on available cache memory.
* :code_link:`/max/api/python/kv_cache/paged_cache/transfer_engine|available_port`: Find an available TCP port for transfer engine communication.

Modules
=======

* :code_link:`/max/api/python/kv_cache/registry|registry`: KV cache manager factory functions and utilities.

Packages
========

* :code_link:`/max/api/python/kv_cache/paged_cache|paged_cache`: Paged attention KV cache implementation.

Classes
=======

* :code_link:`/max/api/python/kv_cache/paged_cache/cache_manager|PagedKVCacheManager`: Manager for paged KV cache with data and tensor parallelism support.
* :code_link:`/max/api/python/kv_cache/paged_cache/transfer_engine|KVTransferEngine`: Manages KV cache transfers between devices in distributed settings.
* :code_link:`/max/api/python/kv_cache/paged_cache/transfer_engine|KVTransferEngineMetadata`: Metadata for KV cache transfer engine configuration.
* :code_link:`/max/api/python/kv_cache/paged_cache/transfer_engine|TransferReqData`: Data structure for KV cache transfer requests.

.. toctree::
   :hidden:

   paged_cache/index
   registry
