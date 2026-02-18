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
"""Helper functions for asserting in ops."""

from collections.abc import Iterable

from ..value import BufferValue, TensorValue


def indent(
    lines: Iterable[str], level: int = 1, indent: str = "    "
) -> Iterable[str]:
    """Prepends each line with the given indent string repeated level times."""
    for line in lines:
        yield (indent * level) + line


def assert_same_device(
    *values: TensorValue | BufferValue,
    **named_values: TensorValue | BufferValue,
) -> None:
    """Raises ValueError if any of the given values are not on the same device."""
    named_values = {
        **{str(i): value for i, value in enumerate(values)},
        **named_values,
    }
    if len({v.device for v in named_values.values()}) > 1:
        raise ValueError(
            "Input values must be on the same device\n"
            + "\n".join(
                indent(
                    f"{name}: {value.type}"
                    for name, value in named_values.items()
                )
            )
        )


def assert_on_host(
    *values: TensorValue | BufferValue,
    **named_values: TensorValue | BufferValue,
) -> None:
    """Raises ValueError if any of the given values are not on the CPU device."""
    named_values = {
        **{str(i): value for i, value in enumerate(values)},
        **named_values,
    }
    not_on_host = {
        k: v for k, v in named_values.items() if not v.device.is_cpu()
    }
    if not_on_host:
        raise ValueError(
            "Input value must be on the host device\n"
            + "\n".join(
                indent(
                    f"{name}: {value.type}"
                    for name, value in not_on_host.items()
                )
            )
        )


def assert_same_shape(*values: TensorValue | BufferValue) -> None:
    """Raises ValueError if the given values do not all have the same shape."""
    first_shape = values[0].shape

    for i, tensor in enumerate(values[1:], start=1):
        if tensor.shape != first_shape:
            raise ValueError(
                f"All input tensors must have the same shape. "
                f"Input 0 has shape {first_shape}, but input {i} has shape {tensor.shape}"
            )


def assert_same_dtype(*values: TensorValue | BufferValue) -> None:
    """Raises ValueError if the given values do not all have the same dtype."""
    first_dtype = values[0].dtype

    for i, tensor in enumerate(values[1:], start=1):
        if tensor.dtype != first_dtype:
            raise ValueError(
                f"All input tensors must have the same dtype. "
                f"Input 0 has dtype {first_dtype}, but input {i} has dtype {tensor.dtype}"
            )


def assert_valid_axis(value: TensorValue | BufferValue, axis: int) -> None:
    """Raises IndexError if axis is out of range for the value's rank."""
    if not (-value.rank <= axis < value.rank):
        raise IndexError(
            f"Axis must be in range [-{value.rank}, {value.rank}), got {axis}"
        )
