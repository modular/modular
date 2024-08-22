:description: The MAX Engine Python API reference.
:title: MAX Engine Python API
:sidebar_label: max.engine

MAX Engine Python API
=====================

.. py:currentmodule:: max.engine

You can run an inference with our Python API in just a few lines of code:

1. Create an :obj:`InferenceSession`.
2. Load a model with :obj:`InferenceSession.load()`, which
   returns a :obj:`Model`.
3. Run the model by passing your input to :obj:`Model.execute()`, which returns
   the output.

That's it! For more detail, see
`how to run inference with Python </max/python/get-started>`_.


.. autoclass:: max.engine.InferenceSession
   :members:
   :undoc-members:

.. autoclass:: max.engine.Model
   :members:
   :undoc-members:

.. autoclass:: max.engine.TensorSpec
   :members:
   :undoc-members:

.. autoclass:: max.engine.TorchInputSpec
   :members:
   :undoc-members:
