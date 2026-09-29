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

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   mesh_context

Placements
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Partial
   Placement
   Replicated
   Sharded

Layouts
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BufferLayout
   TensorLayout

Tensor-to-mesh mappings
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DeviceMapping
   NamedMapping

Per-op decisions
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ActionSet
   AxisAssignment

Resharding
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   auto_reshard

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   ALL_TRANSITIONS
   DEFAULT_TRANSITIONS
   Transition

Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ConversionError
   ShardingError

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_action_set
   force_replicated_action_set
