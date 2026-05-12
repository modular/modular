:title: max.interfaces
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.interfaces
==============

.. automodule:: max.interfaces
   :no-members:

.. currentmodule:: max.interfaces

Pipeline base
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Pipeline
   PipelineInputs
   PipelineOutput
   PipelinesFactory
   PipelineTask
   PipelineTokenizer

Text generation
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BatchType
   MessageContent
   TextContentPart
   TextGenerationContext
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
   EmbeddingsGenerationInputs
   EmbeddingsGenerationOutput

Audio generation
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AudioGenerationInputs
   AudioGenerationMetadata
   AudioGenerationOutput
   AudioGenerationRequest

Image generation
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ImageContentPart
   ImageMetadata
   PixelGenerationContext
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
   GenerationOutput
   GenerationStatus
   SamplingParams
   SamplingParamsGenerationConfigDefaults
   SamplingParamsInput

Requests and scheduling
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   OpenResponsesRequest
   Request
   RequestID
   Scheduler
   SchedulerResult

Tokens
------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   LogProbabilities
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

LoRA
----

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   LoRAOperation
   LoRARequest
   LoRAResponse
   LoRAStatus
   LoRAType

Queues
------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   MAXPullQueue
   MAXPushQueue

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   drain_queue
   get_blocking

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
