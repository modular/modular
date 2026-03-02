:description: The MAX pipelines API reference.
:title: pipelines
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/pipelines/

pipelines
---------

The `pipelines` package provides end-to-end implementations for text
generation, embeddings, audio generation, and speech processing that
automatically convert Hugging Face models into performance-optimized MAX graphs.
Each pipeline can be served via OpenAI-compatible endpoints for production
deployment.

Modules
=======

* :code_link:`/max/api/python/pipelines/architectures|architectures`: Architecture classes for all supported model types.
* :code_link:`/max/api/python/pipelines/config|config`: Pipeline configuration for text generation, embeddings, audio generation, and more.
* :code_link:`/max/api/python/pipelines/core|core`: Core pipeline request and response types.
* :code_link:`/max/api/python/pipelines/hf_utils|hf_utils`: Utilities for interacting with Hugging Face repositories and weight files.
* :code_link:`/max/api/python/pipelines/interfaces|interfaces`: Abstract base classes and protocols for pipeline components.
* :code_link:`/max/api/python/pipelines/log_probabilities|log_probabilities`: Log probability computation graphs.
* :code_link:`/max/api/python/pipelines/lora_config|lora_config`: LoRA adapter configuration and management.
* :code_link:`/max/api/python/pipelines/model_config|model_config`: Model configuration dataclasses.
* :code_link:`/max/api/python/pipelines/pipeline|pipeline`: Pipeline implementations for text generation.
* :code_link:`/max/api/python/pipelines/registry|registry`: Model registry and factory functions.
* :code_link:`/max/api/python/pipelines/sampling|sampling`: Token sampling and speculative decoding utilities.
* :code_link:`/max/api/python/pipelines/tokenizer|tokenizer`: Tokenization utilities for pipeline inputs.


.. toctree::
   :hidden:

   architectures
   config
   core
   hf_utils
   interfaces
   lora_config
   log_probabilities
   model_config
   pipeline
   registry
   sampling
   tokenizer
