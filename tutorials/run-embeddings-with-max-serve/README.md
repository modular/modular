# Embeddings endpoint

A document management system that uses embeddings and clustering to organize and
search through documents semantically.

1. Start the embeddings endpoint:

    ```bash
    max-pipelines serve --model-path=sentence-transformers/all-mpnet-base-v2
    ```

1. Run the system:

    ```bash
    python -m embeddings.kb_system
    ```
