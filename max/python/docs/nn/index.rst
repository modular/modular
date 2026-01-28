:description: The MAX Python Neural Network API reference.
:title: nn
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/nn/

MAX Neural Network API
----------------------

APIs to build neural network components for deep learning models with Python.

The MAX neural network API provides two namespaces:

- **max.nn**: Eager-style execution.
- **max.nn.legacy**: Legacy graph-based API (for backward compatibility).

For functional operations like relu, softmax, and more, see the
:code_link:`/max/api/python/functional|functional` module.

Core API
========

Use these modules for all models. They provide eager-style execution with
PyTorch-style syntax.

* :code_link:`/max/api/python/nn/Embedding|Embedding`: Vector embedding layer for token representation.
* :code_link:`/max/api/python/nn/Linear|Linear`: Linear transformation layer with weights and bias.
* :code_link:`/max/api/python/nn/module|module`: Base class for all neural network modules.
* :code_link:`/max/api/python/nn/norm|norm`: Normalization layers for training stability.
* :code_link:`/max/api/python/nn/rope|rope`: Rotary position embeddings for sequence models.
* :code_link:`/max/api/python/nn/sequential|sequential`: Containers for composing modules sequentially.

Legacy API
==========

.. note::
   The legacy API remains available for backward compatibility. For all new models,
   use the `max.nn` API.

The legacy API provides graph-based layer implementations. See the full
reference:

* :code_link:`/max/api/python/nn/legacy|legacy`: Neural network legacy API documentation.


.. toctree::
   :hidden:

   module
   Linear
   Embedding
   sequential
   norm/index
   rope/index
   legacy/index
