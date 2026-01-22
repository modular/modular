:description: Graduated MAX Python APIs (formerly experimental).
:title: experimental
:card_title: Graduated APIs
:type: package
:lang: python
:wrapper_class: rst-index

max.experimental (Graduated)
----------------------------

.. note::

   The APIs formerly in :obj:`max.experimental` have graduated to top-level
   modules. Please update your imports:

   - ``max.experimental.functional`` → ``max.functional``
   - ``max.experimental.tensor`` → ``max.tensor``
   - ``max.experimental.random`` → ``max.random``

These modules are now stable and ready for production use.

Modules
=======

* :code_link:`/max/api/python/experimental/functional|functional`: Functional APIs for tensor operations (now at ``max.functional``).
* :code_link:`/max/api/python/experimental/random|random`: Random tensor generation utilities (now at ``max.random``).
* :code_link:`/max/api/python/experimental/tensor|tensor`: Tensor operations with eager execution (now at ``max.tensor``).

.. toctree::
   :maxdepth: 2
   :hidden:

   functional
   random
   tensor
