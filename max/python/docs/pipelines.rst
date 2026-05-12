:title: max.pipelines
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines
=============

.. automodule:: max.pipelines
   :no-members:

.. currentmodule:: max.pipelines


Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AudioGenerationConfig
   KVCacheConfig
   LoRAConfig
   MAXModelConfig
   PipelineConfig
   ProfilingConfig
   SamplingConfig
   SpeculativeConfig

Pipelines
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EmbeddingsPipeline
   PixelGenerationPipeline
   SpeechTokenGenerationPipeline
   TextGenerationPipeline
   TextGenerationPipelineInterface

Model interface
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GenerateMixin
   MemoryEstimator
   ModelInputs
   ModelOutputs
   PipelineModel

Context
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   PixelContext
   TextAndVisionContext
   TextContext
   TTSContext

Tokenizers
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   IdentityPipelineTokenizer
   TextAndVisionTokenizer
   TextTokenizer

Enums
-----

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   PipelineRole
   PrometheusMetricsMode
   RepoType
   RopeType
   SupportedEncoding

Utilities
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   PrependPromptSpeechTokens

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   download_weight_files
   is_float4_encoding
   parse_supported_encoding_from_file_name
   supported_encoding_dtype
   supported_encoding_quantization
   supported_encoding_supported_devices
   supported_encoding_supported_on
   upper_bounded_default

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   ADAPTER_CONFIG_FILE

Submodules
----------

.. toctree::
   :maxdepth: 1

   pipelines.architectures
   pipelines.core
   pipelines.lib
   pipelines.lib.interfaces
   pipelines.lib.log_probabilities
   pipelines.lib.registry
