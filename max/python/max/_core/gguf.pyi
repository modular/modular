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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import os
from collections.abc import Sequence

import max._core.driver

class GGUF:
    """A parser for the GGUF (GPT-Generated Unified Format) format."""

    @property
    def version(self) -> int:
        """The GGUF format version of the file."""

    @property
    def alignment(self) -> int:
        """The alignment of the tensor data section, from ``general.alignment``."""

    def keys(self) -> list[str]:
        """Returns the list of tensor keys present."""

    def metadata(self) -> dict[str, str]:
        """
        Returns the file's string-valued metadata. Numbers, booleans and arrays are skipped, as are the tokenizer's vocabulary and merges tables, which are not needed to read tensors.
        """

    def element_shape(self, name: str) -> list[int]:
        """
        Returns the logical element shape of a tensor, before any quantization blocks are taken into account.
        """

    def quantization(self, name: str) -> str | None:
        """
        Returns the name of the tensor's quantization scheme, such as ``"Q4_K"``, or ``None`` when the tensor is not block-quantized.
        """

    def get_buffer(self, name: str) -> max._core.driver.Buffer:
        """Returns a buffer with a given key."""

    def __enter__(self) -> GGUF: ...
    def __exit__(
        self,
        exc_type: object | None = None,
        exc_value: object | None = None,
        traceback: object | None = None,
    ) -> None: ...

class GGUFWriter:
    """A writer for the GGUF (GPT-Generated Unified Format) format."""

    def __init__(self, alignment: int = 32) -> None:
        """
        Creates a writer.

        Metadata and tensors are written in the order they are added. Tensor
        data is copied, so it may be released once added.

        Args:
            alignment: The alignment of the tensor data section, in bytes. It
                must be a power of two, and 32 is the format's default.
        """

    @property
    def alignment(self) -> int:
        """The alignment of the tensor data section."""

    def set_string(self, key: str, value: str) -> None:
        """Adds or replaces a string-valued metadata entry."""

    def set_int(self, key: str, value: int) -> None:
        """
        Adds or replaces a signed integer metadata entry, written as an ``INT64``.
        """

    def set_uint(self, key: str, value: int) -> None:
        """
        Adds or replaces an unsigned integer metadata entry, written as a ``UINT64``.
        """

    def set_float(self, key: str, value: float) -> None:
        """
        Adds or replaces a floating point metadata entry, written as a ``FLOAT64``.
        """

    def set_bool(self, key: str, value: bool) -> None:
        """Adds or replaces a boolean metadata entry."""

    def add_tensor(
        self, name: str, tensor_type: int, shape: Sequence[int], data: bytes
    ) -> None:
        """
        Adds a tensor.

        Args:
            name: The tensor name.
            tensor_type: The GGML type id of `data`; 0 is ``F32``, 30 is
                ``BF16``, 12 is ``Q4_K``.
            shape: The logical shape, widest dimension first.
            data: The tensor data, exactly the bytes the shape and type need.
                For block-quantized types the last dimension counts elements,
                so keep it a whole number of quantization blocks.
        """

    def write(self, filepath: str | os.PathLike) -> None:
        """Writes the file, replacing it if it exists."""

def gguf_open(
    filepath: str | os.PathLike, device: max._core.driver.Device | None = None
) -> GGUF:
    """
    Loads and parses a GGUF file from the given path onto the given device.

    Defaults to loading on the CPU.

    .. note::

       Currently only implemented for CPUs.
    """
