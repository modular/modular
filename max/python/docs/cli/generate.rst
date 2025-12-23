:title: max generate


Generates output from a given model and prompt, without using an
endpoint—primarily for debugging and testing purposes.

For example:

.. code-block:: bash

    max generate \
      --model google/gemma-3-12b-it \
      --max-length 1024 \
      --max-new-tokens 500 \
      --top-k 40 \
      --temperature 0.7 \
      --seed 42 \
      --prompt "Explain quantum computing"


.. raw:: markdown

    :::note

    You can adjust parameters like `--max-batch-size` and `--max-length` depending on
    your system's available resources such as GPU memory.

    :::

    For more information on how to use the `generate` command with vision models,
    see [Image to text](/max/inference/image-to-text).

.. click:: max.entrypoints.pipelines:cli_pipeline
  :prog: max generate
  :hide-description:
