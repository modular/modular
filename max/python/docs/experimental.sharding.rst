:title: max.experimental.sharding
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.experimental.sharding
=========================

.. automodule:: max.experimental.sharding
   :no-members:

.. currentmodule:: max.experimental.sharding

Device mesh
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DeviceMesh

Placements
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Partial
   Placement
   ReduceOp
   Replicated
   Sharded

Sharding specifications
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DeviceMapping
   NamedMapping
   PlacementMapping
   SpecEntry

Distributed types
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DistributedBufferType
   DistributedTensorType
   DistributedType
   TensorLayout

Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ConversionError

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   global_shape_from_local
   local_shard_shape_from_global
   shard_shape
