:title: max serve

Launches a model server with an OpenAI-compatible endpoint. Just specify the
model as a Hugging Face model ID or a local path.

For example:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --devices gpu:0 \
      --max-batch-size 8 \
      --device-memory-utilization 0.9

For details about the endpoint APIs provided by the server, see [the MAX REST
API reference](/max/api/serve).

The ``max`` CLI also supports loading custom model architectures through the
``--custom-architectures`` flag. This allows you to extend MAX's capabilities
with your own model implementations:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --custom-architectures path/to/module1:module1 \
      --custom-architectures path/to/module2:module2


.. raw:: markdown

    :::note Custom architectures

    For a full example using the
    [`--custom-architectures`](#--custom-architectures-custom_architectures) flag,
    see [Serve custom model architectures
    ](/max/develop/serve-custom-model-architectures).

    :::

.. click:: max.entrypoints.pipelines:cli_serve
  :prog: max serve
  :hide-description:
