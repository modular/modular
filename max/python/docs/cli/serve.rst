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

    The `--custom-architectures` flag allows you to load custom pipeline
    architectures from your own Python modules. You can set the `ARCHITECTURES`
    variable containing the architecture definitions. Each entry in
    `--custom-architectures` can be specified in two formats:

    - A raw module name; for example: `my_module`.
    - An import path followed by a colon and the module name; for example: `folder/path/to/import:my_module`.

    The `ARCHITECTURES` variable in your module should be a list of implementations
    that conform to the
    [SupportedArchitecture](/max/api/python/pipelines/registry#max.pipelines.lib.registry.SupportedArchitecture)
    interface. These will be registered with the MAX pipeline registry automatically.

    :::


    :::note Quantization encoding

    When using GGUF models, quantization encoding formats are automatically detected.
    If no `--quantization-encoding` is specified, MAX Serve automatically detects and
    uses the first encoding option from the repository. If quantization encoding is
    provided, it must align with the available encoding options in the repository.

    If the repository contains multiple quantization formats, specify which encoding
    type you want to use with the `--quantization-encoding` parameter.

    :::

.. click:: max.entrypoints.pipelines:cli_serve
  :prog: max serve
  :hide-description:
