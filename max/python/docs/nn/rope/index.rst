:description: The MAX Python rotary positional embeddings API reference.
:title: rope
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/nn/rope/

max.nn.rope
-----------

Rotary positional embedding (RoPE) implementations.

Classes
=======

* :code_link:`/max/api/python/nn/rope/RotaryEmbedding|RotaryEmbedding`: Standard rotary position embedding implementation.
* :code_link:`/max/api/python/nn/rope/TransposedRotaryEmbedding|TransposedRotaryEmbedding`: RoPE with transposed tensor layout.

.. toctree::
   :hidden:

   RotaryEmbedding
   TransposedRotaryEmbedding

.. note::
   For legacy RoPE variants (DynamicRotaryEmbedding, YarnRotaryEmbedding, etc.),
   see :doc:`../legacy/rotary_embedding`.
