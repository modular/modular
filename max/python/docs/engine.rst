:description: The MAX Engine Python API reference.
:title: engine
:type: module
:lang: python
:source: max/engine/

The APIs in this module allow you to run inference with MAX Engine—a graph
compiler and runtime that accelerates your AI models on a wide variety of
hardware.

.. autoclass:: max.engine.InferenceSession
   :members:
   :undoc-members:
   :exclude-members: set_debug_print_options

.. autoclass:: max.engine.Model
   :special-members: __call__
   :members:
   :undoc-members:
   :exclude-members: devices, input_devices, output_devices, signature

.. autoclass:: max.engine.GPUProfilingMode
   :members:
   :undoc-members:

.. autoclass:: max.engine.LogLevel
   :members:
   :undoc-members:

.. autoclass:: max.engine.TensorSpec
   :members:
   :undoc-members:

.. autodata:: max.engine.CustomExtensionsType
