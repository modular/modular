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

* :code_link:`/max/api/python/pipelines/architectures|architectures`
* :code_link:`/max/api/python/pipelines/config|config`
* :code_link:`/max/api/python/pipelines/core|core`
* :code_link:`/max/api/python/pipelines/hf_utils|hf_utils`
* :code_link:`/max/api/python/pipelines/interfaces|interfaces`
* :code_link:`/max/api/python/pipelines/lora_config|lora_config`
* :code_link:`/max/api/python/pipelines/model_config|model_config`
* :code_link:`/max/api/python/pipelines/pipeline|pipeline`
* :code_link:`/max/api/python/pipelines/registry|registry`
* :code_link:`/max/api/python/pipelines/sampling|sampling`
* :code_link:`/max/api/python/pipelines/tokenizer|tokenizer`


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
