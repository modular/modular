:title: embedding
:type: module
:lang: python


The :obj:`embedding` module provides classes for mapping integer indices (like
token IDs) to dense vector representations. These embedding operations are
fundamental building blocks for natural language processing, recommendation
systems, and other tasks involving discrete tokens.


* ``Embedding``: Basic embedding lookup table for simple use cases
* ``EmbeddingV2``: Enhanced embedding with device placement control and improved memory management
* ``VocabParallelEmbedding``: Distributed embedding that shards the vocabulary across multiple devices for large embedding tables


Here's an example demonstrating how to use embeddings:

.. code-block:: python

    import max.nn as nn
    from max.graph import Graph, ops, DeviceRef
    from max.dtype import DType
    import numpy as np

    with Graph(name="embedding_example") as graph:
        # Define dimensions
        batch_size = 4
        seq_length = 16
        vocab_size = 10000
        hidden_dim = 256

        # Create input tensor of token indices
        input_data = np.random.randint(0, vocab_size, (batch_size, seq_length), dtype=np.int32)
        input_indices = ops.constant(input_data, dtype=DType.int32, device=DeviceRef.CPU())

        # Create embedding layer
        embedding = nn.EmbeddingV2(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="token_embeddings"
        )

        # Look up embeddings for input indices
        embeddings = embedding(input_indices)
        print(f"Embedding output shape: {embeddings.shape}")
        # Embedding output shape: [Dim(4), Dim(16), Dim(256)]

.. automodule:: max.nn.embedding
   :members:
   :undoc-members:
