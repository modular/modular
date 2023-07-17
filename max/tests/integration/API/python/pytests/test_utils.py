# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import version_info

if version_info.minor <= 8:
    from typing import List
else:
    List = list

from pathlib import Path

import numpy as np

import modular.engine as me

"""Common testing utilities across testing files"""

# TODO: We may wish to provide such utilities to users.

_batch_size = 1


# TODO: We need better automatic casting; explicit .astype() calls shouldn't be required
def _get_np_dtype(dtype: me.DType):
    if dtype == me.DType.bool:
        return np.bool_
    elif dtype == me.DType.int8:
        return np.int8
    elif dtype == me.DType.int16:
        return np.int16
    elif dtype == me.DType.int32:
        return np.int32
    elif dtype == me.DType.int64:
        return np.int64
    elif dtype == me.DType.uint8:
        return np.uint8
    elif dtype == me.DType.uint16:
        return np.uint16
    elif dtype == me.DType.uint32:
        return np.uint32
    elif dtype == me.DType.uint64:
        return np.uint64
    elif dtype == me.DType.float16:
        return np.float16
    elif dtype == me.DType.float32:
        return np.float32
    elif dtype == me.DType.float64:
        return np.float64


def generate_test_inputs(model: me.Model) -> List[np.ndarray]:
    input_specs = model.input_metadata
    inputs: List[np.ndarray] = []
    for spec in input_specs:
        input_shape_dyn = spec.shape
        input_shape = [
            _batch_size if not dim else dim for dim in input_shape_dyn
        ]
        input_type = spec.dtype
        random_input = np.random.rand(*input_shape).astype(
            _get_np_dtype(input_type)
        )
        inputs.append(random_input)
    return inputs


def compile_and_execute_on_random_input(model_path: Path):
    session = me.InferenceSession()
    model: me.Model = session.load(model_path)
    assert model is not None
    inputs = generate_test_inputs(model)
    output = model.execute(*inputs)
    assert output is not None
