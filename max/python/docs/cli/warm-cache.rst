:title: max warm-cache


Preloads and compiles the model to optimize initialization time by:

- Pre-compiling models before deployment
- Warming up the Hugging Face cache

This command is useful to run before serving a model.

For example:

.. code-block:: bash

    max warm-cache \
      --model google/gemma-3-12b-it

.. raw:: markdown

    :::note

    The Modular Executable Format (MEF) is platform independent, but
    the serialized cache (MEF files) produced during compilation is
    platform-dependent. This is because:

    - Platform-dependent optimizations happen during compilation.
    - Fallback operations assume a particular runtime environment.

    Weight transformations and hashing during MEF caching can impact performance.
    While efforts to improve this through weight externalization are ongoing,
    compiled MEF files remain platform-specific and are not generally portable.

    :::

.. click:: max.entrypoints.pipelines:cli_warm_cache
  :prog: max warm-cache
  :hide-description:
