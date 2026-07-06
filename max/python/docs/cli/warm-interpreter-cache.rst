:title: max warm-interpreter-cache

MAX includes an interpreter that runs operations one at a time as your graph
calls them. For some of these operations, such as matrix multiplication and
elementwise math, the interpreter builds an optimized, compiled version the
first time it runs the operation. It saves each compiled version to an on-disk
cache and reuses it on later runs.

There's one compiled version for each combination of operation, device, and
data type your machine supports, so compiling them all on first use can take
several minutes. Use ``max warm-interpreter-cache`` to compile every combination
for the current hardware up front.

Because the compiled results depend on the hardware, run this command on the
same kind of machine you plan to run on. A common use is during system
provisioning, such as a step in a Dockerfile after you install MAX.

.. raw:: markdown

    :::note

    To let other MAX processes reuse the compiled results, set the
    `MODULAR_DERIVED_PATH` environment variable to the same value those processes
    use before you run this command. MAX saves the cache under that
    location and records your machine's hardware alongside it.

    :::

.. click:: max._entrypoints.pipelines:cli_warm_interpreter_cache
  :prog: max warm-interpreter-cache
  :hide-description:
