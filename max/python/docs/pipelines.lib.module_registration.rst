:title: max.pipelines.lib.module_registration
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.lib.module_registration
=====================================

.. automodule:: max.pipelines.lib.module_registration
   :no-members:

.. currentmodule:: max.pipelines.lib.module_registration

Pipeline models
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ModuleV3DecoderPipelineModel
   RegisteredDecoderBuilder

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   module_pipeline_model

Batch processors
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DecoderBatchProcessor
   VisionBatchProcessor

Model inputs
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DecoderModelInputs

Module construction
-------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   construct_module

Registration accessors
----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   device_mesh
   graph_name
   supports_pipeline_stages

Graph input types
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   decoder_input_types
   decoder_vision_input_types

Workspace and state binding
---------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   bind_decoder_scratch
   bind_state_store
