:description: Experimental MAX Python APIs for advanced users and early adopters.
:title: experimental
:card_title: Experimental APIs
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/experimental/

max.experimental
----------------

.. caution::

   The APIs in this module are experimental and subject to change or removal
   in future releases without prior notice. Use with caution in production
   environments.

The :obj:`max.experimental` package provides experimental APIs for the MAX
platform. These APIs are designed for early adopters who want to explore new
features before they become stable.

Experimental APIs may have:

- Incomplete or changing interfaces.
- Limited documentation or examples.
- Performance characteristics that may change.
- Breaking changes between releases.

Modules
=======

* :code_link:`/max/api/python/experimental/functional|functional`: Functional APIs for tensor operations.
* :code_link:`/max/api/python/experimental/random|random`: Random tensor generation utilities.
* :code_link:`/max/api/python/experimental/tensor|tensor`: Tensor operations with eager execution capabilities.

.. toctree::
   :maxdepth: 2
   :hidden:

   functional
   random
   tensor
