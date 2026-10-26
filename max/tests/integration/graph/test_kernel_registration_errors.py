# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Kernel registration errors point at the Mojo code, like Mojo diagnostics."""

import os
import re
from pathlib import Path

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, KernelRegistrationError, TensorType, ops

# `file.mojo:line:col: error: message`, as the Mojo compiler prints it.
_MOJO_ERROR = r"__init__\.mojo:\d+:\d+: error: "


def _assert_points_at(message: str, source: str) -> None:
    """Asserts that `message` quotes the Mojo line `source` with a caret."""
    assert re.search(r"\n *" + re.escape(source) + r"\n *\^", message), (
        f"expected the source line {source!r} and a caret in:\n{message}"
    )


@pytest.fixture
def bad_kernel_ops_path() -> Path:
    return Path(os.environ["MODULAR_BAD_KERNEL_OPS_PATH"])


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_bad_registrations_fail_at_load(bad_kernel_ops_path: Path) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    with Graph("bad_registrations", input_types=[tensor_type]) as graph:
        graph._import_kernels([bad_kernel_ops_path])
        # The package loads, and is checked, on its first use.
        with pytest.raises(KernelRegistrationError) as err:
            ops.custom(
                "context_not_last",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )

    message = str(err.value)
    assert isinstance(err.value, ValueError)
    # The package imported fine, so only the kernel errors are reported.
    assert "failed to import kernels" not in message
    assert re.search(
        _MOJO_ERROR + "Context argument must be the last argument", message
    )
    assert re.search(
        _MOJO_ERROR + "Outputs must appear before inputs in the function "
        "signature: output 'output' comes after input 'x'",
        message,
    )
    _assert_points_at(message, "ctx: DeviceContext,")
    _assert_points_at(
        message, "output: OutputTensor[dtype=x.dtype, rank=x.rank, ...],"
    )


def test_failed_package_does_not_break_later_loads(
    bad_kernel_ops_path: Path, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    with Graph("after_failure", input_types=[tensor_type]) as graph:
        graph._import_kernels([bad_kernel_ops_path])
        with pytest.raises(KernelRegistrationError):
            "context_not_last" in graph._kernel_library  # noqa: B015

        # The bad package was rolled back, so a good one loads cleanly.
        graph._import_kernels([kernel_verification_ops_path])
        ops.custom(
            "my_add",
            device=DeviceRef.CPU(),
            values=[graph.inputs[0], graph.inputs[0]],
            out_types=[tensor_type],
        )


def test_call_mismatch_points_at_kernel(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    with Graph("call_mismatch", input_types=[tensor_type]) as graph:
        graph._import_kernels([kernel_verification_ops_path])
        with pytest.raises(KernelRegistrationError) as err:
            ops.custom(
                "my_add",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )

    message = str(err.value)
    assert re.search(
        _MOJO_ERROR + "Could not provide all input arguments to op 'my_add'",
        message,
    )
    assert "note: in the 'execute' entry point of 'my_add'" in message
    # The error points at the missing argument, `y`, below the `def execute`
    # line the note points at. (Shipping this package's source in the test's
    # runfiles would shadow its precompiled form, so the line isn't quoted.)
    error_line = re.search(r"__init__\.mojo:(\d+):\d+: error: ", message)
    note_line = re.search(r"__init__\.mojo:(\d+):\d+: note: ", message)
    assert error_line and note_line
    assert int(error_line.group(1)) > int(note_line.group(1))


def test_unknown_kernel_suggests_name(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    with Graph("unknown_kernel", input_types=[tensor_type]) as graph:
        graph._import_kernels([kernel_verification_ops_path])
        with pytest.raises(KernelRegistrationError) as err:
            ops.custom(
                "my_ad",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0], graph.inputs[0]],
                out_types=[tensor_type],
            )

    assert "did you mean 'my_add'?" in str(err.value)
