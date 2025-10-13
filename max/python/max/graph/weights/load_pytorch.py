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
"""Function for importing a PyTorch checkpoint into a MAX Graph."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from typing import Any

import numpy as np
import torch  # type: ignore
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from .weights import WeightData


@dataclass
class TensorInfo:
    dtype: torch.dtype
    offset: int
    shape: tuple[int, ...]


class WeightUnpickler(pickle.Unpickler):
    def __init__(self, pkl_file, zip_file) -> None:  # noqa: ANN001
        super().__init__(pkl_file)
        self.zip_file = zip_file

    def build_tensor(  # noqa: ANN201
        self,
        zip_info,  # noqa: ANN001
        unused_storage_offset,  # noqa: ANN001
        size,  # noqa: ANN001
        *unused_args,
        **unused_kwargs,
    ):
        zip_info.shape = size
        return zip_info

    def find_class(self, module, name):  # noqa: ANN001, ANN201
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self.build_tensor
        return super().find_class(module, name)

    def persistent_load(self, pid):  # noqa: ANN001, ANN201
        data = pid[1:]
        storage_type, key, unused_location, unused_num_elements = data

        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        name = f"data/{key}"
        offset = self.zip_file.get_record_offset(name)
        return TensorInfo(dtype=dtype, offset=offset, shape=())


class PytorchWeights:
    """Implementation for loading weights from PyTorch checkpoint files.

    ``PytorchWeights`` provides an interface to load model weights from PyTorch
    checkpoint files (.bin or .pt format). These files contain serialized
    PyTorch tensors using Python's pickle protocol, making them widely compatible
    with the PyTorch ecosystem.

    .. code-block:: python

        from pathlib import Path
        from max.graph.weights import PytorchWeights
        from max.dtype import DType

        # Load weights from PyTorch checkpoint
        checkpoint_path = Path("pytorch_model.bin")
        weights = PytorchWeights(checkpoint_path)

        # Check if a weight exists before allocation
        if weights.model.decoder.layers[0].self_attn.q_proj.weight.exists():
            # Allocate the attention weight
            q_weight = weights.model.decoder.layers[0].self_attn.q_proj.weight.allocate(
                dtype=DType.float32,
                device=DeviceRef.CPU()
            )

        # Access weight properties
        if weights.embeddings.weight.exists():
            print(f"Embedding shape: {weights.embeddings.weight.shape}")
            print(f"Embedding dtype: {weights.embeddings.weight.dtype}")

        # Allocate with validation
        embedding_weight = weights.embeddings.weight.allocate(
            dtype=DType.float16,
            shape=(50257, 768)  # Validate expected shape
        )
    """

    _filepath: PathLike[str]
    _tensor_infos: dict[str, Any]
    _prefix: str
    _allocated: dict[str, DLPackArray]

    def __init__(
        self,
        filepath: PathLike[str],
        tensor_infos: dict[str, Any] | None = None,
        prefix: str = "",
        allocated=None,  # noqa: ANN001
    ) -> None:
        self._filepath = filepath
        if tensor_infos is not None:
            self._tensor_infos = tensor_infos
        else:
            zip_file = torch._C.PyTorchFileReader(str(filepath))
            with BytesIO(zip_file.get_record("data.pkl")) as pkl_file:
                unpickler = WeightUnpickler(pkl_file, zip_file)
                self._tensor_infos = unpickler.load()
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    @property
    def dtype(self) -> DType:
        """The current weight dtype, if this weight exists."""
        return DType.from_torch(self._tensor_infos[self._prefix].dtype)

    @property
    def shape(self) -> Shape:
        """The current weight shape, if this weight exists."""
        return Shape(self._tensor_infos[self._prefix].shape)

    @property
    def quantization_encoding(self) -> QuantizationEncoding | None:
        """The current weight quantization encoding, if this weight exists."""
        return None

    def items(self):  # noqa: ANN201
        """Iterate through all allocable weights that start with the prefix."""
        for name in self._tensor_infos:
            if name.startswith(self._prefix):
                yield (
                    name,
                    PytorchWeights(
                        self._filepath,
                        tensor_infos=self._tensor_infos,
                        prefix=name,
                        allocated=self._allocated,
                    ),
                )

    def __getattr__(self, attr) -> PytorchWeights:  # noqa: ANN001
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        if not any(name.startswith(full_path) for name in self._tensor_infos):
            raise AttributeError(f"No weight {full_path} found")
        return PytorchWeights(
            self._filepath,
            tensor_infos=self._tensor_infos,
            prefix=full_path,
            allocated=self._allocated,
        )

    def __getitem__(self, idx: int | str) -> PytorchWeights:
        return self.__getattr__(str(idx))

    def data(self) -> WeightData:
        tensor_info = self._tensor_infos[self._prefix]
        dtype = DType.from_torch(self._tensor_infos[self._prefix].dtype)
        return WeightData(
            np.memmap(
                self._filepath,
                mode="r",
                dtype=np.uint8,
                offset=tensor_info.offset,
            ),
            self.name,
            dtype,
            Shape(tensor_info.shape),
        )

    def exists(self) -> bool:
        return self._prefix in self._tensor_infos

    def allocate(
        self,
        dtype: DType | None = None,
        shape: ShapeLike | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> Weight:
        """Creates and optionally validates a new Weight."""
        if quantization_encoding:
            raise ValueError(
                f"Could not load quantized weight {self._prefix} from pytorch"
                " checkpoint:"
            )

        tensor_info = self._tensor_infos[self._prefix]
        weight = Weight(
            self._prefix,
            DType.from_torch(tensor_info.dtype),
            tensor_info.shape,
            device=device,
        )
        self._allocated[self._prefix] = np.memmap(
            self._filepath, mode="r", dtype=np.uint8, offset=tensor_info.offset
        )

        # Validate the loaded weight.
        if shape is not None:
            expected_shape = tuple(shape)
            weight_unpacked_shape = tuple(dim for dim in weight.shape)

        if shape is not None and weight_unpacked_shape != expected_shape:
            raise ValueError(
                f"Value provided to weight '{self.name}' had different shape"
                f" (expected={expected_shape}, actual={weight_unpacked_shape})"
            )
        if dtype is not None and weight.dtype != dtype:
            raise ValueError(
                f"Value provided to weight '{self.name}' had different dtype"
                f" (expected={dtype}, actual={weight.dtype})"
            )

        return weight

    @property
    def allocated_weights(self) -> dict[str, DLPackArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
