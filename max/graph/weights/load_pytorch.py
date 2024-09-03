# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for importing a PyTorch checkpoint into a MAX Graph."""

import pickle
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

try:
    import torch
except ImportError:
    torch = None

from max.dtype import DType

from ..weight import Weight


def _dtype_from_torch(dtype) -> DType:
    torch_to_dtype = {
        torch.bool: DType.bool,
        torch.int8: DType.int8,
        torch.int16: DType.int16,
        torch.int32: DType.int32,
        torch.int64: DType.int64,
        torch.uint8: DType.uint8,
        # torch.uint16: DType.uint16,  # Pytorch doesn't support these uint dtypes.
        # torch.uint32: DType.uint32,
        # torch.uint64: DType.uint64,
        torch.float16: DType.float16,
        torch.float32: DType.float32,
        torch.float64: DType.float64,
        torch.bfloat16: DType.bfloat16,
    }

    return torch_to_dtype[dtype]


def load_pytorch(filepath) -> Dict[str, tuple[Weight, npt.NDArray]]:
    if torch is None:
        raise ImportError(
            "Unable to import torch. Please make sure that PyTorch is installed"
            " on your system."
        )
    zip_file = torch._C.PyTorchFileReader(str(filepath))

    with BytesIO(zip_file.get_record("data.pkl")) as pkl_file:
        unpickler = WeightUnpickler(pkl_file, zip_file)
        loaded_infos = unpickler.load()

    ret: dict[str, tuple[Weight, npt.NDArray]] = {}
    for key, tensor_info in loaded_infos.items():
        dtype = _dtype_from_torch(tensor_info.dtype)
        ret[key] = (
            Weight(key, dtype, tensor_info.shape),
            np.memmap(
                filepath, mode="r", dtype=np.uint8, offset=tensor_info.offset
            ),
        )
    return ret


@dataclass
class TensorInfo:
    dtype: Any  # torch.dtype
    offset: int
    shape: Tuple[int, ...]


class WeightUnpickler(pickle.Unpickler):
    def __init__(self, pkl_file, zip_file):
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
