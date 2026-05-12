:description: The MAX Python API reference.
:title: max
:card_title: Python API
:type: package
:lang: python
:wrapper_class: rst-index
:sidebar_position: 1

max
---

The MAX Python API reference.

The MAX API provides a high-performance graph compiler and runtime library that
executes AI models with incredible speed on a wide range of hardware.

MAX offers a layered architecture that lets you work at the level of abstraction
that best fits your needs. From deploying production-ready models with a few
lines of code to building custom neural networks from scratch, each layer builds
upon the others so you can move between levels seamlessly as requirements evolve.

For an introduction, see the
`Model developer guide </max/develop/>`_.

Modules
=======

.. toctree::
   :maxdepth: 1

   driver
   dtype
   engine
   entrypoints
   graph
   graph.ops
   graph.quantization
   graph.weights
   interfaces
   kv_cache
   nn
   nn.attention
   nn.kernels
   nn.kv_cache
   pipelines
   pipelines.architectures
   pipelines.architectures.bert
   pipelines.architectures.deepseekV2
   pipelines.architectures.deepseekV3
   pipelines.architectures.deepseekV3_2
   pipelines.architectures.deepseekV3_nextn
   pipelines.architectures.eagle3_deepseekV3
   pipelines.architectures.eagle_llama3
   pipelines.architectures.exaone
   pipelines.architectures.exaone_modulev3
   pipelines.architectures.flux2
   pipelines.architectures.flux2_modulev3
   pipelines.architectures.gemma3
   pipelines.architectures.gemma3_modulev3
   pipelines.architectures.gemma3multimodal
   pipelines.architectures.gemma3multimodal_modulev3
   pipelines.architectures.gemma4
   pipelines.architectures.gpt_oss
   pipelines.architectures.gpt_oss_modulev3
   pipelines.architectures.granite
   pipelines.architectures.granite_modulev3
   pipelines.architectures.idefics3
   pipelines.architectures.idefics3_modulev3
   pipelines.architectures.internvl
   pipelines.architectures.kimik2_5
   pipelines.architectures.lfm2
   pipelines.architectures.llama3
   pipelines.architectures.llama3_modulev3
   pipelines.architectures.mamba
   pipelines.architectures.minimax_m2
   pipelines.architectures.mistral
   pipelines.architectures.mistral3
   pipelines.architectures.mpnet
   pipelines.architectures.mpnet_modulev3
   pipelines.architectures.olmo
   pipelines.architectures.olmo2
   pipelines.architectures.olmo2_modulev3
   pipelines.architectures.olmo3
   pipelines.architectures.olmo_modulev3
   pipelines.architectures.phi3
   pipelines.architectures.phi3_modulev3
   pipelines.architectures.pixtral
   pipelines.architectures.pixtral_modulev3
   pipelines.architectures.qwen2
   pipelines.architectures.qwen2_5vl
   pipelines.architectures.qwen3
   pipelines.architectures.qwen3_5
   pipelines.architectures.qwen3_embedding
   pipelines.architectures.qwen3_embedding_modulev3
   pipelines.architectures.qwen3vl_moe
   pipelines.architectures.qwen_image
   pipelines.architectures.qwen_image_edit
   pipelines.architectures.step3p5
   pipelines.architectures.unified_eagle_llama3
   pipelines.architectures.unified_mtp_deepseekV3
   pipelines.architectures.wan
   pipelines.architectures.z_image_modulev3
   pipelines.core
   pipelines.lib
   pipelines.lib.interfaces
   pipelines.lib.log_probabilities
   profiler
   profiler.cpu
   profiler.gpu
   experimental
   experimental.nn
   experimental.nn.norm
   experimental.nn.rope
   experimental.tensor
   experimental.random
   experimental.functional
   experimental.sharding
   experimental.torch
