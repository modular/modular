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
"""Msgpack support for NumPy arrays."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import msgspec
import numpy as np
from max.interfaces.generation import GenerationOutput
from max.interfaces.request.open_responses import (
    OpenResponsesRequest,
    OutputImageContent,
    OutputTextContent,
    ReasoningSummaryContent,
    RefusalContent,
)
from pydantic import BaseModel

from .shared_memory import ndarray_to_shared_memory, open_shm_array

logger = logging.getLogger("max.interfaces")

# Type registry for Pydantic models to avoid importlib
_PYDANTIC_TYPE_REGISTRY: dict[str, type] = {}


def _build_type_registry() -> dict[str, type]:
    """Build the static type registry by importing known Pydantic types.

    This avoids the need for dynamic importlib usage during deserialization.

    Returns:
        Dictionary mapping full type paths to type objects.
    """
    registry: dict[str, type] = {}

    # Register each type with its full module path
    for cls in [
        OpenResponsesRequest,
        OutputImageContent,
        OutputTextContent,
        RefusalContent,
        ReasoningSummaryContent,
        GenerationOutput,
    ]:
        type_key = f"{cls.__module__}.{cls.__qualname__}"
        registry[type_key] = cls

    return registry


# Initialize registry at module load time
_PYDANTIC_TYPE_REGISTRY = _build_type_registry()

SHARED_MEMORY_THRESHOLD = 24000000


def numpy_encoder_hook(
    use_shared_memory: bool = False,
    shared_memory_threshold: int = SHARED_MEMORY_THRESHOLD,
) -> Callable[[Any], Any]:
    """Create a configurable numpy encoding hook.

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
        """Custom encoder that handles numpy arrays and Pydantic models with optional shared memory conversion."""
        # Handle Pydantic BaseModel instances
        if isinstance(obj, BaseModel):
            return {
                "__pydantic__": True,
                "type": obj.__class__.__module__
                + "."
                + obj.__class__.__qualname__,
                "data": obj.model_dump(mode="json"),
            }

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


class MsgpackNumpyEncoder:
    """A pickleable encoder class for msgpack data with numpy arrays.

    This class wraps msgspec.msgpack.Encoder functionality in a pickleable
    container by storing the encoder parameters and recreating the encoder
    as needed.
    """

    def __init__(
        self,
        use_shared_memory: bool = False,
        shared_memory_threshold: int = SHARED_MEMORY_THRESHOLD,
    ):
        """Initialize the encoder.

        Args:
            use_shared_memory: Whether to attempt shared memory conversion for numpy arrays
            shared_memory_threshold: Minimum size in bytes for shared memory conversion.
                                    If 0, all arrays are candidates for conversion.
        """
        if (
            use_shared_memory
            and float(os.environ.get("MODULAR_MAX_SHM_WATERMARK", 0.9)) == 0.0
        ):
            logger.warning(
                "MODULAR_MAX_SHM_WATERMARK is set to 0.0, shared memory will be disabled."
            )
            self._use_shared_memory = False
        else:
            self._use_shared_memory = use_shared_memory

        self._shared_memory_threshold = shared_memory_threshold
        self._encoder: msgspec.msgpack.Encoder | None = None
        self._create_encoder()

    def _create_encoder(self) -> None:
        """Create the internal msgspec encoder."""
        enc_hook = numpy_encoder_hook(
            self._use_shared_memory, self._shared_memory_threshold
        )
        self._encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)

    def __call__(self, obj: Any) -> bytes:
        """Encode object into bytes.

        Args:
            obj: The object to encode

        Returns:
            The encoded bytes
        """
        if self._encoder is None:
            self._create_encoder()

        assert self._encoder is not None
        return self._encoder.encode(obj)

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling (excluding the non-pickleable encoder)."""
        return {
            "_use_shared_memory": self._use_shared_memory,
            "_shared_memory_threshold": self._shared_memory_threshold,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling and recreate the encoder."""
        self._use_shared_memory = state["_use_shared_memory"]
        self._shared_memory_threshold = state["_shared_memory_threshold"]
        self._encoder = None
        self._create_encoder()


def msgpack_numpy_encoder(
    use_shared_memory: bool = False,
    shared_memory_threshold: int = SHARED_MEMORY_THRESHOLD,
) -> MsgpackNumpyEncoder:
    """Create an encoder function that handles numpy arrays.

    Args:
        use_shared_memory: Whether to attempt shared memory conversion for numpy arrays
        shared_memory_threshold: Minimum size in bytes for shared memory conversion.
                                If 0, all arrays are candidates for conversion.

    Returns:
        A pickleable encoder instance that encodes objects into bytes
    """
    return MsgpackNumpyEncoder(use_shared_memory, shared_memory_threshold)


class MsgpackNumpyDecoder:
    """A pickleable decoder class for msgpack data with numpy arrays.

    This class wraps msgspec.msgpack.Decoder functionality in a pickleable
    container by storing the decoder parameters and recreating the decoder
    as needed.
    """

    def __init__(self, type_: Any, copy: bool = False):
        """Initialize the decoder.

        Args:
            type_: The type to decode into
            copy: Whether to copy numpy arrays when deserializing. Defaults to False.
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


def msgpack_numpy_decoder(
    type_: Any, copy: bool = False
) -> MsgpackNumpyDecoder:
    """Create a decoder function for the specified type.

    Args:
        type_: The type to decode into
        copy: Copy numpy arrays if true. Defaults to True.
            Copy is set to True by default because most downstream usage of deserialized tensors are MAX driver tensors, which require owned numpy arrays.
            This is a constraint imposed by dlpack & numpy where we cannot create a buffer from read-only data.
            While there is a performance benefit during deserialization to removing copies by default, this often just moves the work downstream to an implicit copy during `Buffer.from_numpy`.
            As a result, it is easier to make the copy explicit here and maintain the pattern that all numpy arrays used in MAX are owned by the current process.

    Returns:
        A pickleable decoder instance that decodes bytes into the specified type
    """
    return MsgpackNumpyDecoder(type_, copy)


def decode_numpy_array(type_: type, obj: Any, copy: bool) -> Any:
    """Custom decoder for numpy arrays and Pydantic models from msgspec.

    Args:
        type_: The expected type (not used in this implementation)
        obj: The object to decode
        copy: Whether to copy the array data.

    Raises:
        ValueError: If a Pydantic type is not registered in the type registry.
    """
    # Handle Pydantic BaseModel instances
    if isinstance(obj, dict) and obj.get("__pydantic__") is True:
        type_key = obj["type"]

        # Get the class from the registry
        pydantic_class = _PYDANTIC_TYPE_REGISTRY.get(type_key)

        if pydantic_class is None:
            raise ValueError(
                f"Pydantic type '{type_key}' is not registered in the type registry. "
                f"Please add it to _build_type_registry() in "
                f"max/python/max/interfaces/utils/serialization.py. "
                f"Available types: {list(_PYDANTIC_TYPE_REGISTRY.keys())}"
            )

        try:
            # Reconstruct the Pydantic model from the dumped data
            # Type ignore needed because mypy can't infer that registry contains BaseModel subclasses
            return pydantic_class.model_validate(obj["data"])  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Failed to validate Pydantic model data: {e}")
            raise

    if isinstance(obj, dict) and obj.get("__np__") is True:
        # Wrapping the frombuffer in an array to avoid potential issues with data ownership across process boundaries.
        return np.array(
            np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(
                obj["shape"]
            ),
            copy=copy,
        )

    if isinstance(obj, dict) and obj.get("__shm__") is True:
        try:
            return open_shm_array(obj)

        except FileNotFoundError:
            raise

    return obj
