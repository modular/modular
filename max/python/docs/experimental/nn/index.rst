:description: Eager neural network modules for the nn API.
:title: nn
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/experimental/nn/

max.experimental.nn
===================

Eager neural network modules using the ``nn`` API.

Classes
=======

* :code_link:`/max/api/python/experimental/nn/module|Module`: Base class for all eager neural network modules.
* :code_link:`/max/api/python/experimental/nn/Conv2d|Conv2d`: 2D convolution layer.
* :code_link:`/max/api/python/experimental/nn/Embedding|Embedding`: Embedding lookup table.
* :code_link:`/max/api/python/experimental/nn/Linear|Linear`: Linear (fully connected) layer.
* :code_link:`/max/api/python/experimental/nn/sequential|Sequential`: Sequential container of modules.

Subpackages
===========

* :doc:`norm/index`: Normalization layers (RMSNorm, LayerNorm, GroupNorm, GemmaRMSNorm).
* :doc:`rope/index`: Rotary positional embeddings (RotaryEmbedding, TransposedRotaryEmbedding).

.. toctree::
   :hidden:

   module
   Conv2d
   Embedding
   Linear
   sequential
   norm/index
   rope/index
