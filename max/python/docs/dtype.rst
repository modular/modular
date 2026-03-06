:description: The DType class for MAX Python API.
:title: dtype
:type: module
:lang: python
:source: max/dtype/dtype.py

.. py:currentmodule:: max.dtype

Provides data type definitions for tensors in MAX Engine. These data types are
essential for defining the precision and memory layout of tensor data when
working with machine learning models.

This module defines the :class:`DType` enum, which represents all supported tensor
data types in MAX Engine, including:

- Integer types (signed and unsigned): ``int8`` | ``uint8`` | ``int16`` | ``uint16`` | ``int32`` | ``uint32`` | ``int64`` | ``uint64``
- Floating-point types (``float8`` variants): ``float16`` | ``bfloat16`` | ``float32`` | ``float64``
- Boolean type: ``bool``

The module also provides utilities for converting between MAX Engine data types
and `NumPy dtypes <https://numpy.org/doc/stable/user/basics.types.html>`_, making
it easy to interoperate with the NumPy ecosystem.

.. code-block:: python

    import numpy as np
    from max.dtype import DType

    # Create a tensor with float32 data type
    tensor = np.zeros((2, 3), dtype=DType.float32.to_numpy())

    # Convert NumPy dtype to MAX DType
    array = np.ones((4, 4), dtype=np.float16)
    max_dtype = DType.from_numpy(array.dtype)

    # Check properties of data types
    is_float = DType.float32.is_float()  # True
    is_int = DType.int64.is_integral()   # True
    size = DType.float64.size_in_bytes   # 8


.. automodule:: max.dtype
   :imported-members:
   :members:
   :undoc-members:
   :exclude-members: finfo, dtype_extension

.. autoclass:: max.dtype.finfo
   :members:
   :undoc-members: