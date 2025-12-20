:title: max
:sidebar_position: 1

.. raw:: markdown

    The `max` command line tool allows you to create an OpenAI-compatible endpoint
    with a simple `max serve` command. It also includes a command to benchmark your
    endpoint using built-in datasets or your own dataset.

    To install the `max` CLI, install the `modular` package as shown
    in the [install guide](/max/packages#install).


.. click:: max.entrypoints.pipelines:main
  :prog: max
  :hide-description:


.. toctree::
   :hidden:

   benchmark.rst
   encode.rst
   generate.rst
   list.rst
   serve.rst
   warm-cache.rst
