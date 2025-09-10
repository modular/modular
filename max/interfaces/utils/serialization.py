# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Msgpack Support for Numpy Arrays"""

from __future__ import annotations

import functools
from typing import Any, Callable

import msgspec
import numpy as np

from .shared_memory import ndarray_to_shared_memory, open_shm_array


def numpy_encoder_hook(
    use_shared_memory: bool = False,
    shared_memory_threshold: int = 24000000,
) -> Callable[[Any], Any]:
    """
    Create a configurable numpy encoding hook.

    Args:
        use_shared_memory: Whether to attempt shared memory conversion for numpy arrays.
        shared_memory_threshold: Minimum size in bytes for shared memory conversion.
            If 0, all arrays are candidates for conversion.
            The default value is 24MB (24,000,000 bytes), which is chosen based on
            internal micro-benchmarks. These benchmarks indicate that serialization
            using shared memory begins to show a measurable speedup for numpy arrays
            at or above this size, making it a practical default for performance-sensitive
            applications.

    Returns:
        Encoding hook function that handles numpy arrays and optionally converts
        them to shared memory.
    """

    def encode_hook(obj: Any) -> Any:
        """Custom encoder that handles numpy arrays with optional shared memory conversion."""
        if isinstance(obj, np.ndarray):
            # Try shared memory conversion if enabled and array meets threshold
            if (
                use_shared_memory
                and obj.nbytes >= shared_memory_threshold
                and (shm_array := ndarray_to_shared_memory(obj)) is not None
            ):
                return {
                    "__shm__": True,
                    "name": shm_array.name,
                    "shape": shm_array.shape,
                    "dtype": shm_array.dtype,
                }

            # Fall back to regular numpy encoding
            return {
                "__np__": True,
                "data": obj.tobytes(),
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }

        return obj

    return encode_hook


def msgpack_numpy_encoder(
    use_shared_memory: bool = False,
    shared_memory_threshold: int = 0,
) -> Callable[[Any], bytes]:
    """Create an encoder function that handles numpy arrays.

    Args:
        use_shared_memory: Whether to attempt shared memory conversion for numpy arrays
        shared_memory_threshold: Minimum size in bytes for shared memory conversion.
                                If 0, all arrays are candidates for conversion.

    Returns:
        A function that encodes objects into bytes
    """
    enc_hook = numpy_encoder_hook(use_shared_memory, shared_memory_threshold)
    encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
    return encoder.encode


class MsgpackNumpyDecoder:
    """A pickleable decoder class for msgpack data with numpy arrays.

    This class wraps msgspec.msgpack.Decoder functionality in a pickleable
    container by storing the decoder parameters and recreating the decoder
    as needed.
    """

    def __init__(self, type_: Any, copy: bool = True):
        """Initialize the decoder.

        Args:
            type_: The type to decode into
            copy: Copy numpy arrays if true
        """
        self._type = type_
        self._copy = copy
        self._decoder: msgspec.msgpack.Decoder[Any] | None = None
        self._create_decoder()

    def _create_decoder(self) -> None:
        """Create the internal msgspec decoder."""
        self._decoder = msgspec.msgpack.Decoder(
            type=self._type,
            dec_hook=functools.partial(decode_numpy_array, copy=self._copy),
        )

    def __call__(self, data: bytes) -> Any:
        """Decode bytes into the specified type.

        Args:
            data: The bytes to decode

        Returns:
            The decoded object
        """
        if self._decoder is None:
            self._create_decoder()

        assert self._decoder is not None
        return self._decoder.decode(data)

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling (excluding the non-pickleable decoder)."""
        return {
            "_type": self._type,
            "_copy": self._copy,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling and recreate the decoder."""
        self._type = state["_type"]
        self._copy = state["_copy"]
        self._decoder = None
        self._create_decoder()


def msgpack_numpy_decoder(type_: Any, copy: bool = True) -> MsgpackNumpyDecoder:
    """Create a decoder function for the specified type.

    Args:
        type_: The type to decode into
        copy: Copy numpy arrays if true

    Returns:
        A pickleable decoder instance that decodes bytes into the specified type
    """
    return MsgpackNumpyDecoder(type_, copy)


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


def msgpack_eq(a: Any, b: Any) -> bool:
    """
    Compare two msgpack-serializable objects for equality. This should really
    only be used in tests.

    Args:
        a: The first object to compare
        b: The second object to compare
    """
    if not isinstance(b, type(a)):
        return False

    # Get all fields from msgspec
    fields = msgspec.structs.fields(type(a))

    # Compare all attributes
    for field in fields:
        field_name = field.name
        self_val = getattr(a, field_name)
        other_val = getattr(b, field_name)

        # Handle numpy arrays
        if isinstance(self_val, np.ndarray):
            if not np.array_equal(self_val, other_val):
                return False
        # Handle lists
        elif isinstance(self_val, list) or isinstance(self_val, tuple):
            if len(self_val) != len(other_val):
                return False
            for s, o in zip(self_val, other_val):
                if isinstance(s, np.ndarray):
                    if not np.array_equal(s, o):
                        return False
                elif s != o:
                    return False
        # Handle sets
        elif isinstance(self_val, set):
            if self_val != other_val:
                return False
        # Handle all other types
        elif self_val != other_val:
            return False

    return True
