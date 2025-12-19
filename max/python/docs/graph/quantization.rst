:title: quantization
:type: module
:lang: python

APIs to quantize graph tensors.

This package includes a comprehensive set of tools for working with quantized
models in MAX Graph. It defines supported quantization encodings, configuration
parameters that control the quantization process, and block parameter
specifications for different quantization formats.

The module supports various quantization formats including 4-bit, 5-bit, and
6-bit precision with different encoding schemes. It also provides support for
GGUF-compatible formats for interoperability with other frameworks.

.. py:currentmodule:: max.graph.quantization

.. autoclass:: max.graph.quantization.BlockParameters
   :members:
   :undoc-members:

.. autoclass:: max.graph.quantization.QuantizationConfig
   :members:
   :undoc-members:

.. autoclass:: max.graph.quantization.QuantizationEncoding
   :members:
   :undoc-members:
   :exclude-members: Q4_0, Q4_K, Q5_K, Q6_K, GPTQ
