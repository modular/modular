# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test error messages for a custom op with errors."""

from pathlib import Path

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_no_operation_dump(custom_ops_mojopkg: Path) -> None:
    """Check that we don't dump IR on failure to elaborate."""
    with pytest.raises(ValueError) as excinfo:
        InferenceSession().load(
            Graph(
                "elab_error",
                forward=lambda: ops.custom(
                    "fails_to_elaborate",
                    values=[],
                    out_types=[
                        TensorType(
                            DType.int32, shape=[42], device=DeviceRef.CPU()
                        )
                    ],
                ),
                custom_extensions=[custom_ops_mojopkg],
            ),
        )

    error_msg = str(excinfo.value)
    assert all(
        phrase not in error_msg
        for phrase in ("see current operation", "builtin.module")
    ), "found internal MLIR operation dump in user-facing error message"
