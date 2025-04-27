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

For a complete walkthrough, see the tutorial to [Deploy a text embedding model
with an
endpoint](https://docs.modular.com/max/tutorials/run-embeddings-with-max-serve)
