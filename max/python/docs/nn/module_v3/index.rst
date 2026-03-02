:description: Eager neural network modules for the module_v3 API.
:title: module_v3
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/nn/module_v3/

max.nn.module_v3
----------------

Eager neural network modules using the ``module_v3`` API.

Classes
=======

* :code_link:`/max/api/python/nn/module_v3/module|Module`: Base class for all eager neural network modules.
* :code_link:`/max/api/python/nn/module_v3/Conv2d|Conv2d`: 2D convolution layer.
* :code_link:`/max/api/python/nn/module_v3/Embedding|Embedding`: Embedding lookup table.
* :code_link:`/max/api/python/nn/module_v3/Linear|Linear`: Linear (fully connected) layer.
* :code_link:`/max/api/python/nn/module_v3/sequential|Sequential`: Sequential container of modules.

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
