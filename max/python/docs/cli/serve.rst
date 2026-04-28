:title: max serve

Launches a model server with an OpenAI-compatible endpoint. Specify the
model as a Hugging Face model ID or a local path.

For example, start a server for a Gemma 3 model on the first GPU:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --devices gpu:0 \
      --max-batch-size 8 \
      --device-memory-utilization 0.9

For details about the endpoint APIs provided by the server, see [the MAX REST
API reference](/max/api/serve).

You can extend MAX with your own model implementations by loading custom
architectures through the ``--custom-architectures`` flag. Each value takes
the form ``path/to/module:module_name``:

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
