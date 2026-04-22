:title: max encode

Converts input text into embeddings for semantic search, text similarity, and
NLP applications.

For example:

.. code-block:: bash

    max encode \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --prompt "Convert this text into embeddings"

.. click:: max.entrypoints.pipelines:encode
  :prog: max encode
  :hide-description: