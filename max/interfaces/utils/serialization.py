# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Msgpack Support for Numpy Arrays"""

from __future__ import annotations

import copy
import functools
from typing import Any, Callable, TypeVar

import msgspec
import numpy as np

from .context import TextAndVisionContext
from .shared_memory import (
    SharedMemoryArray,
    ndarray_to_shared_memory,
    open_shm_array,
)

T = TypeVar("T")


def msgpack_numpy_encoder() -> Callable[[Any], bytes]:
    """Create an encoder function that handles numpy arrays.

    Returns:
        A function that encodes objects into bytes
    """
    encoder = msgspec.msgpack.Encoder(enc_hook=encode_numpy_array)
    return encoder.encode


def _shared_memory(
    pixel_values: tuple[np.ndarray, ...],
) -> tuple[SharedMemoryArray, ...] | None:
    """Convert pixel values to shared memory arrays.

    Args:
        pixel_values: Tuple of numpy arrays to convert

    Returns:
        Tuple of SharedMemoryArray objects if all arrays were successfully
        converted, None if any conversion failed or if input is empty.
    """
    if not pixel_values:
        # Empty tuple doesn't need conversion
        return None

    # Check if there's actually data to convert
    total_bytes = sum(img.nbytes for img in pixel_values)
    if total_bytes == 0:
        return None

    # Try to convert each image to shared memory
    shm_arrays = []
    for img in pixel_values:
        shm_array = ndarray_to_shared_memory(img)
        if shm_array is None:
            # Conversion failed, bail out early
            return None
        shm_arrays.append(shm_array)

    return tuple(shm_arrays)


class SharedMemoryEncoder:
    """Encoder that converts vision contexts to use shared memory.

    This encoder wraps the standard msgpack encoder and adds special handling
    for vision contexts with pixel_values, converting large numpy arrays to
    shared memory for zero-copy transfer between processes.
    """

    def __init__(self) -> None:
        """Initializes the encoder with a base msgpack encoder."""
        self._base_encoder = msgpack_numpy_encoder()

    def __call__(self, obj: Any) -> bytes:
        """Encodes an object to bytes, with special handling for vision contexts.

        Args:
            obj: The object to encode. If it's a (req_id, context) tuple where
                context has pixel_values, attempts to use shared memory.

        Returns:
            Encoded bytes representation of the object.
        """
        # Only process tuples of (req_id, context).
        if not (isinstance(obj, tuple) and len(obj) == 2):
            return self._base_encoder(obj)

        req_id, context = obj

        # Check if context is a vision context.
        if not isinstance(context, TextAndVisionContext):
            return self._base_encoder(obj)

        # Only process if pixel_values is non-empty.
        if not context.pixel_values:
            return self._base_encoder(obj)

        # Try to convert pixel values to shared memory.
        if (shm_arrays := _shared_memory(context.pixel_values)) is not None:
            # Create a shallow copy with shared memory references.
            context_copy = copy.copy(context)
            # Type ignore needed because we're replacing np.ndarray with
            # SharedMemoryArray.
            # However, this assignment is safe because the serialization
            # handles it correctly.
            context_copy.pixel_values = shm_arrays  # type: ignore[assignment]
            obj = (req_id, context_copy)

        return self._base_encoder(obj)


def msgpack_numpy_decoder(
    type_: Any, copy: bool = True
) -> Callable[[bytes], Any]:
    """Create a decoder function for the specified type.

    Args:
        type_: The type to decode into
        copy: Copy numpy arrays if true

    Returns:
        A function that decodes bytes into the specified type
    """
    decoder = msgspec.msgpack.Decoder(
        type=type_, dec_hook=functools.partial(decode_numpy_array, copy=copy)
    )
    return decoder.decode


def encode_numpy_array(obj: Any) -> dict:
    """Custom encoder for numpy arrays to be used with msgspec."""
    if isinstance(obj, np.ndarray):
        return {
            "__np__": True,
            "data": obj.tobytes(),
            "shape": obj.shape,
            "dtype": str(obj.dtype),
        }

    # Handle SharedMemoryArray objects from max.pipelines.core.shared_memory.
    if hasattr(obj, "name") and hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            "__shm__": True,
            "name": obj.name,
            "shape": obj.shape,
            "dtype": obj.dtype,
        }

    return obj


def decode_numpy_array(type_: type, obj: Any, copy: bool) -> Any:
    """Custom decoder for numpy arrays from msgspec.

    Args:
        type_: The expected type (not used in this implementation)
        obj: The object to decode
        copy: Whether to copy the array data.
    """
    if isinstance(obj, dict) and obj.get("__np__") is True:
        arr = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(
            obj["shape"]
        )

        if copy:
            arr = np.copy(arr)

        return arr

    if isinstance(obj, dict) and obj.get("__shm__") is True:
        try:
            return open_shm_array(obj)

        except FileNotFoundError:
            raise

    return obj
