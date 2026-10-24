:title: max.pipelines.lib.interfaces
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.lib.interfaces
============================

.. automodule:: max.pipelines.lib.interfaces
   :no-members:

.. currentmodule:: max.pipelines.lib.interfaces

Architecture config
-------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ArchConfig
   ArchConfigWithAttentionKVCache
   ArchConfigWithBoundedMaxSeqLen
   ArchConfigWithKVCache
   ArchConfigWithPermissiveMaxSeqLen
   ArchConfigWithStoredKVParams
   ArchConfigWithVisionCache
   ArchVLConfigWithTextSubconfig

Batch processing
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BatchProcessor
   BatchProcessorRuntime
   ModuleV3SingleReplicaBatchProcessor
   RaggedBatchProcessor
   SingleReplicaRaggedBatchProcessor

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   arch_has_vision_tower
   build_single_replica_ragged_token_arrays
   modulev3_ragged_kv_symbolic_input_tree
   ragged_batch_inputs
   ragged_kv_symbolic_input_tree
   signal_inputs
   single_replica_context_batch
   vision_merge_inputs
