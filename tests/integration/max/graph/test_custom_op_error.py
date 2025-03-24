# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test error messages for a custom op with errors."""

import re
from pathlib import Path

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops


def test_no_operation_dump(custom_ops_mojopkg: Path) -> None:
    """Check that we don't dump IR on failure to elaborate."""
    with pytest.raises(ValueError) as excinfo:
        InferenceSession().load(
            Graph(
                "elab_error",
                forward=lambda: ops.custom(
                    "fails_to_elaborate",
                    values=[],
                    out_types=[TensorType(DType.int32, shape=[42])],
                ),
            ),
            custom_ops_path=str(custom_ops_mojopkg),
        )

    # Verify two conditions:
    # 1. Our expected error message is present.
    # 2. MLIR's "see current operation" notes are filtered out.
    error_msg = str(excinfo.value)
    assert re.search(
        r"user_invalid\.mojo:.* error: call expansion failed.*note: constraint failed: oops",
        error_msg,
        re.DOTALL,
    ), "missing expected error pattern"
    assert all(
        phrase not in error_msg
        for phrase in ("see current operation", "builtin.module")
    ), "found internal MLIR operation dump in user-facing error message"
