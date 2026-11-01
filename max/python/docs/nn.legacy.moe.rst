:title: max.nn.legacy.moe
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn.legacy.moe
=================

.. automodule:: max.nn.legacy.moe
   :no-members:

.. currentmodule:: max.nn.legacy.moe

Mixture of experts
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GatedActivation
   SigmoidTopKRouter
   StackedMoE

Quantization strategies
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Fp8Strategy
   GateUpFormat
   Nvfp4Scales
   NvMxf4f8Strategy
   QuantStrategy

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   interleaved_block_scales_shape
   make_concatenated_gated_activation_fn
   make_interleaved_gated_activation_fn
   make_stacked_gated_activation_fn
