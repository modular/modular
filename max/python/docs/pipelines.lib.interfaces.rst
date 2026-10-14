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
   RaggedBatchProcessor

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   arch_has_vision_tower
   modulev3_ragged_kv_symbolic_input_tree
   ragged_batch_inputs
   ragged_kv_symbolic_input_tree
   signal_inputs
   vision_merge_inputs
