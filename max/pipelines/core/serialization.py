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

import functools
from typing import Any, Callable, TypeVar

import msgspec
import numpy as np

T = TypeVar("T")


def msgpack_numpy_encoder() -> Callable[[Any], bytes]:
    """Create an encoder function that handles numpy arrays.

    Returns:
        A function that encodes objects into bytes
    """
    encoder = msgspec.msgpack.Encoder(enc_hook=encode_numpy_array)
    return encoder.encode


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

    return obj


def decode_numpy_array(type_: type, obj: Any, copy: bool) -> Any:
    """Custom decoder for numpy arrays from msgspec.

    Args:
        type_: The expected type (not used in this implementation)
        obj: The object to decode
    """
    if isinstance(obj, dict) and obj.get("__np__") is True:
        arr = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(
            obj["shape"]
        )

        if copy:
            arr = np.copy(arr)

        return arr

    return obj
