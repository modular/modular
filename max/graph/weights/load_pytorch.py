# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for importing a PyTorch checkpoint into a MAX Graph."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from max.graph import DeviceRef

try:
    import torch  # type: ignore
except ImportError:
    torch = None

from max.dtype import DType

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from .weights import WeightData


@dataclass
class TensorInfo:
    dtype: Any  # torch.dtype
    offset: int
    shape: tuple[int, ...]


class WeightUnpickler(pickle.Unpickler):
    def __init__(self, pkl_file, zip_file) -> None:
        super().__init__(pkl_file)
        self.zip_file = zip_file

    def build_tensor(
        self,
        zip_info,
        unused_storage_offset,
        size,
        *unused_args,
        **unused_kwargs,
    ):
        zip_info.shape = size
        return zip_info

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self.build_tensor
        return super().find_class(module, name)

    def persistent_load(self, pid):
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
    _filepath: PathLike
    _tensor_infos: dict[str, Any]
    _prefix: str
    _allocated: dict[str, np.ndarray]

    def __init__(
        self,
        filepath: PathLike,
        tensor_infos: Optional[dict[str, Any]] = None,
        prefix: str = "",
        allocated=None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "Unable to import torch. Please make sure that PyTorch is"
                " installed on your system."
            )
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
    def quantization_encoding(self) -> Optional[QuantizationEncoding]:
        """The current weight quantization encoding, if this weight exists."""
        return None

    def items(self):
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

    def __getattr__(self, attr) -> PytorchWeights:
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

    def raw_tensor(self) -> npt.NDArray[Any]:
        """Returns the tensor corresponding to this weights object.

        Raises:
            KeyError if this weights object isn't a tensor.
        """
        if self._prefix not in self._tensor_infos:
            raise KeyError(
                f"Could not find weight named {self._prefix}. Please check that"
                " the name is correct."
            )

        return self._tensor_infos[self._prefix]

    def data(self) -> WeightData:
        assert torch is not None
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
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
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
            msg = (
                f"Value provided to weight '{self.name}' had different shape"
                f" (expected={expected_shape}, actual={weight_unpacked_shape})"
            )
            raise ValueError(msg)
        if dtype is not None and weight.dtype != dtype:
            msg = (
                f"Value provided to weight '{self.name}' had different dtype"
                f" (expected={dtype}, actual={weight.dtype})"
            )
            raise ValueError(msg)

        return weight

    @property
    def allocated_weights(self) -> dict[str, npt.NDArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
