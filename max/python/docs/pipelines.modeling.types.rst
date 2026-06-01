:title: max.pipelines.modeling.types
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.modeling.types
============================

.. automodule:: max.pipelines.modeling.types
   :no-members:

.. currentmodule:: max.pipelines.modeling.types

Submodules
----------

.. toctree::
   :maxdepth: 1

   pipelines.modeling.types.pipeline_variants

Pipeline base
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   InputModality
   Pipeline
   PipelineInputs
   PipelineInputsType
   PipelineOutput
   PipelineOutputType
   PipelineOutputsDict
   PipelinesFactory
   PipelineTask
   PipelineTokenizer
   TokenizerEncoded
   UnboundContextType

Text generation
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BatchType
   MessageContent
   SpecDecodingState
   TextContentPart
   TextGenerationContext
   TextGenerationContextType
   TextGenerationInputs
   TextGenerationOutput
   TextGenerationRequest
   TextGenerationRequestFunction
   TextGenerationRequestMessage
   TextGenerationRequestTool
   TextGenerationResponseFormat
   VLMTextGenerationContext

Embeddings
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EmbeddingsContext
   EmbeddingsGenerationContextType
   EmbeddingsGenerationInputs
   EmbeddingsGenerationOutput

Image generation
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ImageContentPart
   ImageMetadata
   PixelGenerationContext
   PixelGenerationContextType
   PixelGenerationInputs
   VideoContentPart

Reasoning
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ParsedReasoningDelta
   ReasoningParser
   ReasoningSpan

Tool parsing
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ParsedToolCall
   ParsedToolCallDelta
   ParsedToolResponse
   ToolParser

Context and sampling
--------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BaseContext
   BaseContextType
   EOSTracker
   GenerationOutput
   GenerationStatus
   SamplingParams
   SamplingParamsGenerationConfigDefaults
   SamplingParamsInput

Requests
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   OpenResponsesRequest
   Request
   RequestID
   RequestType

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   DUMMY_REQUEST_ID

Tokens
------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   LogProbabilities
   Range
   TokenBuffer
   TokenSlice

Logit processors
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BatchLogitsProcessor
   BatchProcessorInputs
   LogitsProcessor
   ProcessorInputs

Utilities
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   SharedMemoryArray

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   msgpack_numpy_decoder
   msgpack_numpy_encoder

