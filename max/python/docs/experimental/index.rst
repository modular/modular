:description: Experimental eager-execution Python APIs.
:title: experimental
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/experimental/

max.experimental
----------------

The ``max.experimental`` package contains eager-execution APIs available for
experimentation. These APIs may change in future releases.

* :code_link:`/max/api/python/experimental/functional|functional`: Functional tensor operations (relu, softmax, and more).
* :code_link:`/max/api/python/experimental/nn|nn`: Eager neural network modules (Linear, Module, Embedding, and more).
* :code_link:`/max/api/python/experimental/random|random`: Random tensor generation utilities.
* :code_link:`/max/api/python/experimental/tensor|tensor`: Tensor class with eager execution.
* :code_link:`/max/api/python/experimental/torch|torch`: Custom PyTorch operation integration utilities.

.. toctree::
   :hidden:

   tensor
   functional
   nn/index
   nn/module
   nn/Conv2d
   nn/Linear
   nn/Embedding
   nn/sequential
   nn/norm/index
   nn/rope/index
   random
   torch
