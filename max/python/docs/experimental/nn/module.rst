:description: Module base class for PyTorch-style neural network development.
:title: module
:type: module
:lang: python
:source: max/experimental/nn/module.py

.. py:currentmodule:: max.experimental.nn

The :class:`~max.experimental.nn.module.Module` base class supports a
PyTorch-style workflow for building neural network modules. Write a
``forward()`` method and call :meth:`~max.experimental.nn.module.Module.compile`
to build the MAX graph automatically.

For existing pipeline code or explicit graph control, use
:class:`~max.nn.layer.Module`, which works directly inside a
``with Graph(...)`` block.

.. automodule:: max.experimental.nn.module
   :members:
   :undoc-members:
