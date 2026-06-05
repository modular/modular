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

Per-op decisions
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Action
   ActionSet
   AxisAssignment
   PerShard

Pickers
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GreedyReshard
   NoReshard
   PartialsOnly
   ReshardBehavior
   Solver

Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ConversionError
   ShardingError

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   P
   R

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_action_set
   force_replicated_action_set
   isolated_solver
   mode
