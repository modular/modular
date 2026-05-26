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

The endpoints exposed depend on the value of the ``MAX_SERVE_API_TYPES``
environment variable (default: ``openai,sagemaker``):

- ``openai``: ``/v1/completions``, ``/v1/chat/completions``,
  ``/v1/embeddings``, ``/v1/audio/speech``, ``/v1/models``, ``/v1/health``
- ``sagemaker``: SageMaker-compatible inference endpoints
- ``kserve``: KServe-compatible inference endpoints
- ``responses``: ``/v1/responses`` (required for the ``pixel_generation``
  task)

The OpenAI routes are always registered when the ``openai`` API type is
enabled, but each only functions when the model is served with a
compatible ``--task`` value (for example ``embeddings_generation`` for
``/v1/embeddings``, or ``audio_generation`` for ``/v1/audio/speech``).

To run inference without an HTTP server, see
`max generate </max/cli/generate>`_ (text completion) or
`max encode </max/cli/encode>`_ (embeddings).

For details about the endpoint APIs provided by the server, see [the MAX REST
API reference](/max/rest-api/).

Running on multiple GPUs
------------------------

Pass a comma-separated list of GPU IDs to ``--devices``:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --devices=gpu:0,1,2,3 \
      --max-batch-size 16

``--devices`` is the first-class device selector for ``max serve``.
Avoid combining it with the shell-level ``CUDA_VISIBLE_DEVICES``
environment variable — the two are translated independently and
stacking them can produce wrong device routing under multi-process
workspaces.

Loading non-safetensors checkpoints
-----------------------------------

Some older Hugging Face repositories ship only ``pytorch_model.bin``
and no ``.safetensors`` files. Serving them directly fails at startup
with:

.. code-block:: text

    compatible weights cannot be found for '<encoding>'

The metadata probe never sees a ``.safetensors`` file. To serve these
checkpoints, convert the ``.bin`` to ``model.safetensors`` locally and
point ``--weight-path`` at the converted file:

.. code-block:: python

    import torch
    from safetensors.torch import save_file

    save_file(torch.load("pytorch_model.bin"), "model.safetensors")

Then serve from the converted file:

.. code-block:: bash

    max serve \
      --weight-path ./model.safetensors \
      --quantization-encoding float32 \
      --model-path my-org/my-model

The ``--quantization-encoding`` value must match the converted file's
dtype (typically ``float32`` or ``bfloat16``).

For merged LoRA fine-tunes that already ship in safetensors format,
see `Serve a merged fine-tune
</max/develop/max-pipeline-bring-your-own-model#serve-a-merged-fine-tune>`_
instead.

Common errors
-------------

``ERROR: [Errno 98] address already in use ('0.0.0.0', 8001)``
    The metrics server (default port 8001, controlled by
    ``MAX_SERVE_METRICS_ENDPOINT_PORT``) failed to bind because
    another ``max serve`` process is already using that port. The
    error is benign — only the metrics endpoint fails to bind on the
    second process; the API server continues to serve normally. To
    avoid the warning, set ``MAX_SERVE_METRICS_ENDPOINT_PORT`` to a
    free port on the second process.

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
