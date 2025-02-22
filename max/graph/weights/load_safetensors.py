# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence, Set
from os import PathLike
from typing import Any, Optional

import numpy.typing as npt
from max.dtype import DType

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from ._torch_dtype_map import modular_to_torch_type, torch_to_modular_type
from .weights import WeightData, Weights

try:
    from safetensors import safe_open  # type: ignore
except ImportError:
    safe_open = None

try:
    import torch  # type: ignore
except ImportError:
    torch = None


class SafetensorWeights(Weights):
    """Helper for loading weights into a graph.

    A weight (`max.graph.Weight`) is tensors in a graph which are backed by an
    external buffer or mmap. Generally weights are used to avoid recompiling
    the graph when new weights are used (like from finetuning). For large-enough
    constants, it might be worth using weights for fast compilation times but
    the graph may be less optimized.

    `Weight` classes can be used to help with graph weight allocation and
    naming. This protocol defines getter methods `__getattr__` and `__getitem__`
    to assist with defining names. For example, `weights.a.b[1].c.allocate(...)`
    creates a weight with the name "a.b.1.c".
    """

    _filepaths: Sequence[PathLike]
    _tensors: Set[str]
    _tensors_to_file_idx: Mapping[str, int]
    _allocated: dict[str, npt.NDArray]
    _st_weight_map: dict[str, "torch.Tensor"]

    def __init__(
        self,
        filepaths: Sequence[PathLike],
        *,
        tensors: Optional[Set[str]] = None,
        tensors_to_file_idx: Mapping[str, int] | None = None,
        prefix: str = "",
        allocated=None,
        _st_weight_map: dict[str, "torch.Tensor"] | None = None,
    ):
        if safe_open is None:
            raise ImportError(
                "Could not import safetensors package. Please install it with"
                " `pip install safetensors`."
            )
        # TODO(MSDK-1199): Torch is required in order to import bfloat16.
        if torch is None:
            raise ImportError(
                "Unable to import torch. Please make sure that PyTorch is"
                " installed on your system."
            )
        self._filepaths = filepaths
        if tensors is not None:
            self._tensors = tensors
            assert tensors_to_file_idx is not None
            self._tensors_to_file_idx = tensors_to_file_idx
        else:
            self._tensors_to_file_idx = {}
            self._tensors = set()
            for idx, filepath in enumerate(self._filepaths):
                with safe_open(filepath, framework="numpy") as f:
                    self._tensors |= set(f.keys())
                    self._tensors_to_file_idx |= {k: idx for k in f.keys()}
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated
        self._st_weight_map = {} if _st_weight_map is None else _st_weight_map

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    def items(self):
        """Iterate through all allocable weights that start with the prefix."""
        for name in self._tensors:
            if name.startswith(self.name):
                yield (
                    name,
                    SafetensorWeights(
                        self._filepaths,
                        tensors=self._tensors,
                        tensors_to_file_idx=self._tensors_to_file_idx,
                        prefix=name,
                        allocated=self._allocated,
                        _st_weight_map=self._st_weight_map,
                    ),
                )

    def __getattr__(self, attr) -> SafetensorWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        return SafetensorWeights(
            self._filepaths,
            tensors=self._tensors,
            tensors_to_file_idx=self._tensors_to_file_idx,
            prefix=full_path,
            allocated=self._allocated,
            _st_weight_map=self._st_weight_map,
        )

    def __getitem__(self, idx: int | str) -> SafetensorWeights:
        return self.__getattr__(str(idx))

    def _load_tensor(self, dtype: DType | None = None):
        if self._prefix in self._st_weight_map:
            return self._st_weight_map[self._prefix]

        if self.name not in self._tensors_to_file_idx:
            msg = f"'{self.name}' is not a weight in the Safetensor ckpt."
            if possible_match := difflib.get_close_matches(
                self.name, self._tensors_to_file_idx.keys(), n=1
            ):
                msg += f" Did you mean '{possible_match[0]}'?"
            raise KeyError(msg)

        filepath = self._filepaths[self._tensors_to_file_idx[self.name]]
        assert safe_open is not None
        with safe_open(filepath, framework="pt") as f:
            tensor = f.get_tensor(self.name)

        # Some checkpoints have mixed bf16/float32 weights, while others have
        # weights that are all the same precision.
        # The max graph expects a certain dtype so make sure to convert it here.
        if dtype is not None and torch_to_modular_type(tensor.dtype) != dtype:
            tensor = tensor.to(modular_to_torch_type(dtype))

        self._st_weight_map[self._prefix] = tensor
        return tensor

    def raw_tensor(self) -> npt.NDArray[Any]:
        """Returns the numpy tensor corresponding to this weights object.

        Raises:
            KeyError if this weights object isn't a tensor.
        """
        assert torch is not None
        tensor = self._load_tensor()
        if tensor.dtype == torch.bfloat16:
            np_array = tensor.view(torch.float16).numpy()
        else:
            np_array = tensor.numpy()
        return np_array

    def data(self) -> WeightData:
        assert torch is not None
        tensor = self._load_tensor()
        if tensor.dtype == torch.bfloat16:
            np_array = tensor.view(torch.float16).numpy()
        else:
            np_array = tensor.numpy()
        return WeightData(
            np_array,
            self.name,
            torch_to_modular_type(tensor.dtype),
            Shape(tensor.shape),
        )

    def exists(self) -> bool:
        return self.name in self._tensors_to_file_idx

    def allocate(
        self,
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
    ) -> Weight:
        """Creates a Weight that can be added to a graph."""
        assert torch is not None
        if quantization_encoding is not None:
            raise ValueError(
                "Quantization encodings are not supported in safetensor"
                f" format. Got: {quantization_encoding}"
            )
        tensor = self._load_tensor(dtype)
        weight_dtype = torch_to_modular_type(tensor.dtype)
        if tensor.dtype == torch.bfloat16:
            np_tensor = tensor.view(torch.float16).numpy()
            weight_dtype = DType.bfloat16
        elif tensor.dtype == torch.float8_e4m3fn:
            np_tensor = tensor.view(torch.uint8).numpy()
            weight_dtype = DType.float8_e4m3fn
        elif tensor.dtype == torch.float8_e5m2:
            np_tensor = tensor.view(torch.uint8).numpy()
            weight_dtype = DType.float8_e5m2
        else:
            np_tensor = tensor.numpy()

        weight = Weight(
            name=self._prefix,
            dtype=weight_dtype,
            shape=tensor.shape,
        )
        self._allocated[self._prefix] = np_tensor

        # Validate the loaded weight.
        weight_shape = tuple(dim for dim in weight.shape)
        if shape is not None and weight_shape != tuple(shape):
            raise ValueError(
                f"Did not get expected shape for weight {self._prefix}"
                f"\n\tExpected shape: {shape}, got: {weight_shape}"
            )
        return weight

    def allocate_as_bytes(self, dtype: DType | None = None) -> Weight:
        """Create a Weight that can be added to the graph. Has a uint8
        representation, instead of the original data type. Last dimension of
        the scale gets scaled by number of bytes it takes to represent the
        original data type. For example, [512, 256] float32 weights become
        [512, 1024] uint8 weights. Scalar weights will be interpreted as
        weights with shape [1]."""
        assert torch is not None
        tensor = self._load_tensor(dtype)
        if tensor.ndim == 0:
            tensor = tensor.view([1])
        tensor = tensor.view(torch.uint8)
        np_tensor = tensor.numpy()
        weight_dtype = DType.from_numpy(np_tensor.dtype)

        weight = Weight(
            name=self._prefix, dtype=weight_dtype, shape=tensor.shape
        )
        self._allocated[self._prefix] = np_tensor
        return weight

    @property
    def allocated_weights(self) -> dict[str, npt.NDArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
