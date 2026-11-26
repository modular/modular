:title: max.nn.ep
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn.ep
========================

.. automodule:: max.nn.ep
   :no-members:

.. currentmodule:: max.nn.ep

Configuration and buffers
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EPBuffers
   EPConfig

Exchange
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Grouped
   Route

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   combine
   combine_finish
   combine_local
   combine_start
   dispatch
   dispatch_finish
   dispatch_local
   dispatch_start
   grouped_matmul_bounds

Fused expert FFN
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   mega_ffn_combine_send
   mega_ffn_fused

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   check_supported
