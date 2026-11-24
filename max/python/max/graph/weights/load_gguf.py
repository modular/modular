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
"""Function for importing a GGUF checkpoint into a MAX Graph."""

from __future__ import annotations

import difflib
from collections.abc import Set as AbstractSet
from os import PathLike

from max._core.gguf import GGUF, gguf_open
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import DeviceRef

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from .weights import WeightData, Weights


class GGUFWeights(Weights):
    """Implementation for loading weights from GGUF (GPT-Generated Unified Format) files.

    ``GGUFWeights`` provides an interface to load model weights from GGUF files,
    which are optimized for quantized large language models. GGUF is the
    successor to GGML format and is commonly used in the ``llama.cpp`` ecosystem
    for efficient storage and loading of quantized models.

    .. code-block:: python

        import tempfile
        from pathlib import Path

        import numpy as np
        from max._core.gguf import GGUFWriter
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.graph.weights import GGUFWeights

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            weight = np.ones((4, 4), dtype=np.float32)
            writer = GGUFWriter()
            writer.set_string("general.architecture", "example")
            # 0 is the GGML type id of float32.
            writer.add_tensor(
                "model.embed_tokens.weight", 0, [4, 4], weight.tobytes()
            )
            writer.write(path)

            weights = GGUFWeights(path)

            if weights.model.embed_tokens.weight.exists():
                embedding_weight = weights.model.embed_tokens.weight.allocate(
                    dtype=DType.float32,
                    device=DeviceRef.CPU(),
                )

    .. invisible-code-block: python

        assert embedding_weight.shape == [4, 4]

    GGUF checkpoints can also store block-quantized tensors, such as
    ``Q4_K``. Those are exposed as ``uint8`` data whose shape counts
    quantization blocks, and the weight's ``quantization_encoding`` names the
    scheme, so ``allocate(quantization_encoding=...)`` can validate that a
    layer was stored the way the model expects.
    """

    _reader: GGUF
    _tensors: AbstractSet[str]
    _sorted_tensors: list[str]
    _prefix: str
    _allocated: dict[str, DLPackArray]
    _gguf_weight_map: dict[str, Buffer]

    def __init__(
        self,
        source: PathLike[str] | GGUF,
        tensors: AbstractSet[str] | None = None,
        prefix: str = "",
        allocated: dict[str, DLPackArray] | None = None,
        _gguf_weight_map: dict[str, Buffer] | None = None,
        _sorted_tensors: list[str] | None = None,
    ) -> None:
        """Creates a GGUF weights reader.

        Args:
            source: Path to a GGUF file or a parsed ``GGUF`` reader.
            tensors: Names of every tensor in the checkpoint.
            prefix: Weight name or prefix.
            allocated: Dictionary of allocated values.
            _gguf_weight_map: Buffers already handed out, shared by prefix
                clones so each tensor is decoded at most once.
            _sorted_tensors: Names in declaration order, shared by prefix
                clones.
        """
        self._reader = source if isinstance(source, GGUF) else gguf_open(source)
        self._tensors = (
            frozenset(self._reader.keys()) if tensors is None else tensors
        )
        # Sorted because the reader hands out its keys in hash order, and a
        # graph built by walking these weights declares them in this order.
        self._sorted_tensors = (
            sorted(self._tensors)
            if _sorted_tensors is None
            else _sorted_tensors
        )
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated
        self._gguf_weight_map = (
            {} if _gguf_weight_map is None else _gguf_weight_map
        )

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    def items(self):  # noqa: ANN201
        """Iterate through all allocable weights that start with the prefix."""
        for name in self._sorted_tensors:
            if name.startswith(self._prefix):
                yield (
                    name,
                    GGUFWeights(
                        self._reader,
                        self._tensors,
                        prefix=name,
                        allocated=self._allocated,
                        _gguf_weight_map=self._gguf_weight_map,
                        _sorted_tensors=self._sorted_tensors,
                    ),
                )

    def __getattr__(self, attr: str) -> GGUFWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        return GGUFWeights(
            self._reader,
            self._tensors,
            prefix=full_path,
            allocated=self._allocated,
            _gguf_weight_map=self._gguf_weight_map,
            _sorted_tensors=self._sorted_tensors,
        )

    def __getitem__(self, idx: int | str) -> GGUFWeights:
        return self.__getattr__(str(idx))

    def _validate_name(self) -> str:
        if self._prefix in self._tensors:
            return self._prefix
        possible_match = difflib.get_close_matches(
            self._prefix, self._tensors, n=1
        )
        hint = f" Did you mean '{possible_match[0]}'?" if possible_match else ""
        raise KeyError(
            f"Could not find weight named {self._prefix}. Please check that the"
            f" name is correct.{hint}"
        )

    def _load_buffer(self) -> Buffer:
        """Returns this weight's data, decoding it at most once."""
        name = self._validate_name()
        if name not in self._gguf_weight_map:
            self._gguf_weight_map[name] = self._reader.get_buffer(name)
        return self._gguf_weight_map[name]

    def _quantization_encoding(self) -> QuantizationEncoding | None:
        scheme = self._reader.quantization(self._validate_name())
        return None if scheme is None else QuantizationEncoding(scheme)

    def data(self) -> WeightData:
        """Loads and returns the weight data for this tensor."""
        tensor = self._load_buffer()
        return WeightData(
            tensor,
            self.name,
            tensor.dtype,
            Shape(tensor.shape),
            self._quantization_encoding(),
        )

    def exists(self) -> bool:
        """Returns True if a tensor exists for the current prefix."""
        return self._prefix in self._tensors

    def allocate(
        self,
        dtype: DType | None = None,
        shape: ShapeLike | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> Weight:
        """Creates and optionally validates a new Weight."""
        name = self._validate_name()
        weight = Weight(
            name=name,
            dtype=(tensor := self._load_buffer()).dtype,
            shape=tensor.shape,
            quantization_encoding=self._quantization_encoding(),
            align=self._reader.alignment,
            device=device,
        )
        self._allocated[self._prefix] = tensor

        # Validate the loaded weight. GGUF stores block-quantized tensors as
        # bytes, so compare against the tensor's logical element shape.
        if shape is not None:
            expected_shape = tuple(shape)
            element_shape = tuple(self._reader.element_shape(name))
            if element_shape != expected_shape:
                raise ValueError(
                    f"Value provided to weight '{self.name}' had different"
                    f" shape (expected={expected_shape},"
                    f" actual={element_shape})"
                )
        if dtype is not None and weight.dtype != dtype:
            raise ValueError(
                f"Value provided to weight '{self.name}' had different dtype"
                f" (expected={dtype}, actual={weight.dtype})"
            )

        # GGUF checkpoints can be mixed precision, so we don't check if the
        # quantization encoding matches exactly.
        if quantization_encoding and not weight.quantization_encoding:
            raise ValueError(
                "Expected quantized weight but checkpoint contained"
                f" unquantized weight for {self._prefix}"
            )

        return weight

    @property
    def allocated_weights(self) -> dict[str, DLPackArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
