:title: max.pipelines.lib
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.lib
=================

.. automodule:: max.pipelines.lib
   :no-members:

.. currentmodule:: max.pipelines.lib

Submodules
----------

.. autosummary::
   :nosignatures:

   max.pipelines.lib.arch_lookup
   max.pipelines.lib.interfaces
   max.pipelines.lib.log_probabilities
   max.pipelines.lib.pipeline_parallel

.. toctree::
   :maxdepth: 1
   :hidden:

   pipelines.lib.arch_lookup
   pipelines.lib.auto_processor
   pipelines.lib.generate
   pipelines.lib.generate_images
   pipelines.lib.interfaces
   pipelines.lib.log_probabilities
   pipelines.lib.pipeline_parallel

Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DenoisingCacheConfig
   DenoisingCacheSettings
   KVConnectorConfig
   MAXConfig
   MAXModelConfigBase
   PipelineArgs
   PipelineRuntimeConfig

Pipelines
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EmbeddingsPipelineType
   OverlapTextGenerationPipeline

Model interface
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AlwaysSignalBuffersMixin
   GraphPipelineModelWithKVCache
   ModuleV3MultiGraphPipelineModelWithKVCache
   ModuleV3Outputs
   ModuleV3PipelineModelWithKVCache
   PipelineModelWithKVCache
   SupportsSSMStateWarmup
   UnifiedEagleOutputs
   UnifiedSpecDecodeInputs

Tokenizers
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AudioGenerationTokenizer
   PixelGenerationTokenizer

Utilities
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   CompilationTimer
   HuggingFaceRepo
   MemoryPlan
   ModelManifest
   RetrievedPipeline
   Speculator
   VisionPreprocessCache
   WeightPathParser

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_eos_tracker_for_request
   convert_max_config_value
   deep_merge_max_configs
   generate_local_model_path
   get_default_max_config_file_section_name
   max_tokens_to_generate
   parse_quant_config
   process_ragged_kv_outputs
   ragged_kv_symbolic_inputs
   resolve_eos_token_ids
   resolve_max_config_inheritance
   try_to_load_from_cache
   validate_hf_repo_access

Expert load balancing
---------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.pipelines.lib.eplb_stats.EplbStatsAccumulator
   ~max.pipelines.lib.eplb_stats.EplbStatsMetadata
   ~max.pipelines.lib.eplb_stats.EplbStatsSnapshot

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.pipelines.lib.eplb_rebalance.balanced_packing
   ~max.pipelines.lib.eplb_rebalance.rebalance_experts
   ~max.pipelines.lib.eplb_rebalance.rebalance_experts_hierarchical
   ~max.pipelines.lib.eplb_rebalance.replicate_experts

Output parsers
--------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.pipelines.lib.reasoning.get_parser_cls
   ~max.pipelines.lib.tool_parsing.get_parser_cls

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   ~max.pipelines.lib.config.DEFAULT_MAX_BATCH_INPUT_TOKENS
