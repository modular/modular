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

Modules
=======

.. toctree::
   :maxdepth: 1

   diagnostics.gpu
   driver
   dtype
   engine
   entrypoints
   graph
   graph.ops
   graph.quantization
   graph.weights
   interfaces
   kv_cache
   nn
   nn.attention
   nn.kernels
   nn.kv_cache
   pipelines
   pipelines.core
   pipelines.lib
   pipelines.lib.interfaces
   pipelines.lib.log_probabilities
   profiler
   experimental
   experimental.nn
   experimental.nn.norm
   experimental.nn.rope
   experimental.tensor
   experimental.random
   experimental.functional
   experimental.torch
