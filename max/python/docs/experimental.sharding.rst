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
   Replicated
   Sharded
   Unknown

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

Sharding rules
--------------

A sharding rule receives the op's arguments with each tensor replaced by its
:class:`TensorLayout`, and returns the :class:`AxisAssignment` rows the op
accepts: what each tensor operand must hold and what the op produces. These
functions build common rules and rows.

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AxisAssignment

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   match_operand_placement
   replicated_rows

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

Visualization
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   visualize

Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ConversionError
   ShardingError
