:title: architectures
:type: module
:lang: python
:source: max/pipelines/architectures/

Architecture classes exposed here follow a consistent naming convention: when both
a graph-based and an eager (module_v3) implementation exist for the same model,
the eager implementation uses the ``_ModuleV3`` suffix (e.g. ``SmolLM_ModuleV3``).
The graph-based implementation has no suffix.

.. automodule:: max.pipelines.architectures
   :members:
   :undoc-members: