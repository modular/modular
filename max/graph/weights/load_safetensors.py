# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from os import PathLike
from typing import Optional

import numpy.typing as npt
from max.dtype import DType

from ..quantization import QuantizationEncoding
from ..type import ShapeLike
from ..weight import Weight

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    import torch
except ImportError:
    torch = None


class SafetensorWeights:
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

    def __init__(
        self,
        filepath: PathLike,
        tensors: Optional[set[str]] = None,
        prefix: str = "",
        allocated=None,
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
        self._filepath = filepath
        if tensors is not None:
            self._tensors = tensors
        else:
            with safe_open(filepath, framework="numpy") as f:
                self._tensors = set(f.keys())
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    def items(self):
        """Iterate through allocable weights that start with the weight name."""
        for name in self._tensors:
            if name.startswith(self._prefix):
                yield name, SafetensorWeights(
                    self._filepath,
                    self._tensors,
                    prefix=name,
                    allocated=self._allocated,
                )

    def __getattr__(self, attr) -> SafetensorWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        if not any(name.startswith(full_path) for name in self._tensors):
            raise AttributeError(f"No weight {full_path} found")
        return SafetensorWeights(
            self._filepath,
            self._tensors,
            prefix=full_path,
            allocated=self._allocated,
        )

    def __getitem__(self, idx: int) -> SafetensorWeights:
        return self.__getattr__(str(idx))

    def allocate(
        self,
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
    ) -> Weight:
        """Creates a Weight that can be added to a graph."""
        if quantization_encoding is not None:
            raise ValueError(
                "Quantization encodings are not supported in safetensor"
                f" format. Got: {quantization_encoding}"
            )
        with safe_open(self._filepath, framework="pt") as f:
            tensor = f.get_tensor(self._prefix)
        if tensor.dtype == torch.bfloat16:
            np_tensor = tensor.view(torch.float16).numpy()
            weight_dtype = DType.bfloat16
        else:
            np_tensor = tensor.numpy()
            weight_dtype = DType.from_numpy(np_tensor.dtype)

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
        if dtype is not None and dtype != weight.dtype:
            raise ValueError(
                f"Did not get expected dtype for weight {self._prefix}"
                f"\n\tExpected dtype: {dtype}, got: {weight.dtype}"
            )
        return weight

    @property
    def allocated_weights(self) -> dict[str, npt.NDArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
