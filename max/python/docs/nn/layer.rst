:description: Base classes for building graph-based neural network layers.
:title: layer
:type: module
:lang: python
:source: max/nn/layer/

Base classes for building graph-based neural network layers. Use these
classes directly inside a ``with Graph(...)`` block.

:class:`~max.nn.layer.LayerList` is an ordered container for composing
multiple layers. The :class:`~max.nn.layer.Layer` base class is deprecated.
For new model development, use :class:`~max.experimental.nn.module.Module`
instead.

.. automodule:: max.nn.layer
   :members:
   :undoc-members:
