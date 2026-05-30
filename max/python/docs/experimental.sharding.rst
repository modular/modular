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

Per-shard dim wrappers
----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   PerShardDim

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

   cell_at
   default_mesh
   global_dim
   global_shape
   global_shape_from_local
   is_fully_replicated
   is_one
   is_partial
   is_per_shard_dim
   is_replicated
   is_sharded
   local_shard_shape_from_global
   make_per_shard_dim
   remap_sharded
   replicate_all
   replicate_axes
   resolve_partials
   resolve_partials_mapping
   shape_at
   shard_shape
   sharded_symbolic_dim
