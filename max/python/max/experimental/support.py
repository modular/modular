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

from __future__ import annotations

import contextlib
from collections.abc import Generator
from contextvars import ContextVar
from typing import TypeVar

from max import driver
from max.graph import DeviceRef, TensorType

T = TypeVar("T")


@contextlib.contextmanager
def contextvar_context(var: ContextVar[T], value: T) -> Generator[T]:
    """Context manager that temporarily sets a context variable's value.

    Sets the context variable to the specified value for the duration of the
    context, then resets it to the previous value when the context exits.
    This is useful for scoped configuration changes.

    Args:
        var: The context variable to temporarily modify.
        value: The value to set for the duration of the context.

    Yields:
        The value that was set.

    Example::

        _MY_VAR: ContextVar[int] = ContextVar("_MY_VAR")

        with contextvar_context(_MY_VAR, 42):
            assert _MY_VAR.get() == 42
        # _MY_VAR is now reset to its previous value
    """
    token = var.set(value)
    try:
        yield value
    finally:
        var.reset(token)


def driver_tensor_type(t: driver.Buffer) -> TensorType:
    """Converts a driver tensor to a :obj:TensorType.

    Creates a TensorType instance from a driver-level tensor by extracting
    its dtype, shape, and device information.

    Args:
        t: The driver tensor to convert.

    Returns:
        TensorType: A tensor type representing the driver tensor's properties.
    """
    return TensorType(t.dtype, t.shape, DeviceRef.from_device(t.device))


def driver_tensor_of_type(t: TensorType) -> driver.Buffer:
    """Creates a driver buffer matching the given tensor type."""
    return driver.Buffer(
        t.dtype, [int(d) for d in t.shape], t.device.to_device()
    )
