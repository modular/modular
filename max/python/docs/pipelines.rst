:title: max.pipelines
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines
=============

.. automodule:: max.pipelines
   :no-members:

.. currentmodule:: max.pipelines

Submodules
----------

.. autosummary::
   :nosignatures:

   max.pipelines.architectures
   max.pipelines.audio
   max.pipelines.context
   max.pipelines.diffusion
   max.pipelines.kv_cache
   max.pipelines.lib
   max.pipelines.lib.arch_lookup
   max.pipelines.lib.interfaces
   max.pipelines.lib.log_probabilities
   max.pipelines.lib.registry
   max.pipelines.modeling.base
   max.pipelines.modeling.dataprocessing
   max.pipelines.modeling.types
   max.pipelines.weights
   max.pipelines.lora
   max.pipelines.request
   max.pipelines.sampling
   max.pipelines.speculative

.. toctree::
   :maxdepth: 1
   :hidden:

   pipelines.architectures
   pipelines.audio
   pipelines.context
   pipelines.diffusion
   pipelines.kv_cache
   pipelines.lib
   pipelines.lib.arch_lookup
   pipelines.lib.auto_processor
   pipelines.lib.generate
   pipelines.lib.generate_images
   pipelines.lib.interfaces
   pipelines.lib.log_probabilities
   pipelines.lib.registry
   pipelines.modeling.base
   pipelines.modeling.dataprocessing
   pipelines.modeling.types
   pipelines.weights
   pipelines.lora
   pipelines.request
   pipelines.sampling
   pipelines.speculative

Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   KVCacheConfig
   MAXModelConfig
   PipelineArgs
   PipelineConfig
   PipelineRuntimeConfig
   ProfilingConfig
   SamplingConfig
   SpeculativeConfig

In-process generation
---------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AutoModel
   AutoPipeline
   AutoProcessor
   AutoTokenizer

Pipelines
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EmbeddingsPipeline
   PixelGenerationPipeline
   TextGenerationPipeline
   TextGenerationPipelineInterface

Model interface
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   MemoryEstimator
   ModelInputs
   ModelOutputs
   PipelineModel

Eager usage validation
----------------------

The eager usage validator adds guard rails around eager code running
inside a compiled pipeline. Configure it with the
``runtime.eager_usage_validator`` MAX config setting.

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EagerUsageValidator
   EagerValidatorMode
   ~max.pipelines.modeling.eager_validation.Phase

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.pipelines.modeling.eager_validation.eager_usage_validator

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   eager_validator

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

