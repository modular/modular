:description: The MAX Python Graph API reference.
:title: graph
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/graph/

MAX Graph API
-------------

APIs to build inference graphs for MAX Engine with Python.

Classes
=======

* :code_link:`/max/api/python/graph/BufferValue|BufferValue`: Represents a mutable semantic tensor within a `Graph`.
* :code_link:`/max/api/python/graph/Graph|Graph`: Represents a graph for MAX Engine.
* :code_link:`/max/api/python/graph/KernelLibrary|KernelLibrary`: Represents a library with custom ops.
* :code_link:`/max/api/python/graph/TensorValue|TensorValue`: Represents a value semantic tensor within a `Graph`.
* :code_link:`/max/api/python/graph/Value|Value`: Represents a symbolic value within a `Graph`.
* :code_link:`/max/api/python/graph/Weight|Weight`: Represents a weight value in a graph.


Modules
=======

* :code_link:`/max/api/python/graph/dim|dim`: APIs for graph value tensor dimensions.
* :code_link:`/max/api/python/graph/ops|ops`: Ops you can add when staging a graph.
* :code_link:`/max/api/python/graph/quantization|quantization`: APIs to quantize graph tensors.
* :code_link:`/max/api/python/graph/shape|shape`: APIs for graph value tensor shapes.
* :code_link:`/max/api/python/graph/type|type`: APIs for graph value types.
* :code_link:`/max/api/python/graph/weights|weights`: APIs for loading weights into a graph.


.. toctree::
   :hidden:

   Graph
   KernelLibrary
   BufferValue
   TensorValue
   Value
   Weight
   weights
   dim
   ops
   quantization
   shape
   type
