:description: The MAX Python API reference.
:title: max
:card_title: Python API
:type: package
:lang: python
:wrapper_class: rst-index
:sidebar_position: 1

max
---

The MAX Python API reference.

The MAX API provides a high-performance graph compiler and runtime library that
executes AI models with incredible speed on a wide range of hardware.

MAX offers a layered architecture that lets you work at the level of abstraction
that best fits your needs. From deploying production-ready models with a few
lines of code to building custom neural networks from scratch, each layer builds
upon the others so you can move between levels seamlessly as requirements evolve.

For an introduction, see the
`Model developer guide </max/develop/>`_.

Packages and modules
====================

* :code_link:`/max/api/python/diagnostics/gpu|diagnostics.gpu`: GPU monitoring and performance diagnostics utilities.
* :code_link:`/max/api/python/driver|driver`: Low-level device management and tensor operations.
* :code_link:`/max/api/python/dtype|dtype`: Unified data type system supporting various numeric formats.
* :code_link:`/max/api/python/engine|engine`: Model execution runtime with automatic optimization.
* :code_link:`/max/api/python/entrypoints|entrypoints`: Command-line tools and serving infrastructure.
* :code_link:`/max/api/python/experimental|experimental`: Experimental eager-execution APIs: tensor, functional, and random.
* :code_link:`/max/api/python/graph|graph`: Computational graph construction with 100+ operations for complete model control.
* :code_link:`/max/api/python/interfaces|interfaces`: Universal interfaces for consistent API integration.
* :code_link:`/max/api/python/kv_cache|kv_cache`: KV cache management for efficient attention computation.
* :code_link:`/max/api/python/nn|nn`: Graph-based neural network API (default). Eager API available in ``max.nn.module_v3``.
* :code_link:`/max/api/python/pipelines|pipelines`: Pre-built, optimized model architectures for immediate deployment.
* :code_link:`/max/api/python/profiler|profiler`: Performance profiling and tracing utilities.
* :code_link:`/max/api/python/torch|torch`: PyTorch integration for custom operations and interoperability.

.. toctree::
   :hidden:

   diagnostics/gpu/index.rst
   driver
   dtype
   engine
   entrypoints
   experimental/index.rst
   interfaces
   kv_cache/index.rst
   profiler
   torch
   graph/index.rst
   pipelines/index.rst
   nn/index.rst
