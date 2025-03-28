# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Boilerplate utilities for verifying model output
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing
import yaml
from modular.utils.misc import modular_dtype_to_np_dtype


def _read_tensor_from_file(file_path, shape_dtype_str):
    if "x" in shape_dtype_str:
        shape, dtype = shape_dtype_str.rsplit("x", 1)
    else:
        # for cases of scalar strings e.g. f32 vs in the above branch we might
        # expect 1xf32 (which is a rank-1 tensor)
        shape = "1"
        dtype = shape_dtype_str
    shape = tuple(map(int, shape.split("x")))

    # Convert dtype string to numpy dtype
    if dtype in ["bf16", "bfloat16"]:
        # Since NumPy doesn't have bfloat16 support, convert to float32 in two
        # steps:
        # 1. Read in the buffer as int16.
        with open(file_path, "rb") as f:
            tensor_int16 = np.frombuffer(f.read(), dtype=np.int16)

        # 2. Taking advantage of the bfloat16 format, load the exponent and
        #    mantissa in the upper 16 bits and leave the lower 16 bits of
        #    mantissa as zero.
        #    Note that this only works on little endian systems.
        tensor = np.ascontiguousarray(
            np.stack((np.zeros_like(tensor_int16), tensor_int16)).transpose()
        ).view(np.float32)
    else:
        dtype = modular_dtype_to_np_dtype(dtype)

        # Read binary data and convert to numpy array
        with open(file_path, "rb") as f:
            tensor = np.frombuffer(f.read(), dtype=dtype)

    return tensor.reshape(shape)


def is_close(
    a: numpy.typing.NDArray,
    b: numpy.typing.NDArray,
    absolute_tolerance: float,
    relative_tolerance: float,
    equal_nan: bool,
):
    """Checks if the two input values are numerically within a tolerance.

    When the type is integral, then equality is checked. When the type is
    floating point, then this checks if the two input values are numerically the
    close using the $abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)$
    formula.

    Unlike Pythons's `math.isclose`, this implementation is symmetric. I.e.
    `isclose(a,b) == isclose(b,a)`.

    Args:
      a: The first value to compare.
      a: The second value to compare.
      absolute_tolerance: The absolute tolerance.
      relative_tolerance: The relative tolerance.
      equal_nan: Whether to compare NaN's as equal.

    Returns:
      A boolean vector where a and b are equal within the specified tolerance.
    """
    finite = np.isfinite(a) & np.isfinite(b)

    result = np.zeros_like(finite)
    result[finite] = (
        np.abs(a - b)
        <= np.maximum(
            absolute_tolerance,
            relative_tolerance * np.maximum(np.abs(a), np.abs(b)),
        )
    )[finite]
    # Explicit check for equality of infinite values.
    result[~finite] = a[~finite] == b[~finite]

    if equal_nan:
        both_nan = np.isnan(a) & np.isnan(b)
        result[both_nan] = both_nan[both_nan]

    return result


# Extracts named tensors from the result descriptions.
def _flatten_result_descriptions(result: Any) -> dict:
    d = dict()
    if isinstance(result, dict):
        for k, v in result.items():
            if k == "name" and "shape" in result:
                d[v] = result["shape"]
            d.update(_flatten_result_descriptions(v))
    if isinstance(result, list):
        for v in result:
            d.update(_flatten_result_descriptions(v))
    return d


# Rebuilds the result descriptions such that they become comparable even if the
# order of results is not identical.
def _unify_result_descriptions(
    result_descriptions: list, compare_result_by_position: bool = False
) -> dict:
    results = dict()
    for i, desc in enumerate(result_descriptions):
        if compare_result_by_position:
            desc["name"] = f"result{i}"
        results[desc["name"]] = desc
    return results


# Parse the binary output. `compare_result_by_position` changes the
# result names such that outputs are compared by position rather than name
# (required for modular-pytorch which does not store result names).
def parse_binary_output(
    output_dir: Path, compare_result_by_position: bool = False
) -> tuple[dict, list]:
    with open(output_dir / "output.yaml") as output_yaml:
        result_descriptions = yaml.safe_load(output_yaml)

    # If there is only one result, we may as well compare by position.
    # TODO(#26721): Note that this is currently required because for some models
    # the output names do not match.
    compare_result_by_position = (
        compare_result_by_position or len(result_descriptions) == 1
    )

    tensor_results = dict()
    for desc in result_descriptions:
        tensor_results.update(_flatten_result_descriptions(desc))

    results = dict()
    for name, shape in tensor_results.items():
        illegal_chars = '\\/:?"<>|'
        sanitized_name = "".join(
            c if c not in illegal_chars else "_" for c in name
        )
        tensor_path = output_dir / sanitized_name
        tensor = _read_tensor_from_file(tensor_path, shape)
        results[name] = tensor
    # Return the descriptions, and outputs in an alphabetically sorted order.
    return _unify_result_descriptions(
        result_descriptions, compare_result_by_position
    ), [
        results[name]
        for name in (results if compare_result_by_position else sorted(results))
    ]
