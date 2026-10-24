:title: max.experimental.nn
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.experimental.nn
===================

.. automodule:: max.experimental.nn
   :no-members:

.. currentmodule:: max.experimental.nn

Submodules
----------

.. autosummary::
   :nosignatures:

   max.experimental.nn.norm
   max.experimental.nn.rope

.. toctree::
   :maxdepth: 1
   :hidden:

   experimental.nn.norm
   experimental.nn.rope

Base classes
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Module
   ModuleList
   PinnedDeviceTensor
   Sequential
   TransparentModule

Linear layers
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Conv2d
   Embedding
   Linear

Attention and rotary embeddings
-------------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.experimental.nn.common_layers.attention.AttentionWithRope
   ~max.experimental.nn.common_layers.rotary_embedding.RotaryEmbedding
   ~max.experimental.nn.common_layers.rotary_embedding.YarnRotaryEmbedding

LoRA
----

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   LoRA

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   lora_layers
   lora_parameters

Hyper-connection
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   HyperConnection

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   as_subgraph
   module_dataclass

Mesh axes
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   ~max.experimental.nn.common_layers.mesh_axis.DP
   ~max.experimental.nn.common_layers.mesh_axis.TP
