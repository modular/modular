# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy.typing as npt

try:
    import torch  # type: ignore
except ImportError:
    torch = None
from max.dtype import DType, max_to_torch_type

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight

_Self = TypeVar("_Self", bound="Weights")


@runtime_checkable
class Weights(Protocol):
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

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        ...

    def __getattr__(self: _Self, attr) -> _Self: ...

    def __getitem__(self: _Self, idx: int | str) -> _Self: ...

    def exists(self) -> bool:
        "Returns whether a weight with this exact name exists."
        ...

    def items(self: _Self) -> Iterator[tuple[str, _Self]]:
        """Iterate through all allocable weights that start with the prefix."""
        ...

    def raw_tensor(self) -> npt.NDArray[Any]:
        """Returns the numpy tensor corresponding to this weights object.

        Args:
            dtype: If specified, the returned array will be cast to the dtype
                before returning.
        Raises:
            KeyError if this weights object isn't a tensor.
        """
        ...

    def data(self) -> WeightData:
        """Returns data loaded from the weights at the current prefix.

        Raises:
            KeyError if the current prefix isn't present in the checkpoint.
        """
        ...

    def allocate(
        self,
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
    ) -> Weight:
        """Creates a Weight that can be added to a graph."""
        ...

    @property
    def allocated_weights(self) -> dict[str, npt.NDArray]:
        """Gets the values of all weights that were allocated previously."""
        ...


@dataclasses.dataclass
class WeightData:
    """Data loaded from a checkpoint."""

    data: npt.NDArray
    name: str
    dtype: DType
    shape: Shape
    quantization_encoding: Optional[QuantizationEncoding] = None

    def __dlpack__(self) -> Any:
        return self.data.__dlpack__

    def __dlpack_device__(self) -> Any:
        return self.data.__dlpack_device__

    @classmethod
    def from_numpy(cls, arr, name):
        return cls(arr, name, DType.from_numpy(arr.dtype), Shape(arr.shape))

    def astype(self, dtype: DType) -> WeightData:
        if self.dtype == dtype:
            return self
        if self.dtype == DType.bfloat16:
            assert torch is not None
            data = torch.from_numpy(self.data).view(torch.bfloat16)
            data = data.to(max_to_torch_type(dtype)).numpy()
        elif dtype == DType.bfloat16:
            assert torch is not None
            data = torch.from_numpy(self.data).view(
                max_to_torch_type(self.dtype)
            )
            data = data.to(torch.bfloat16).view(torch.float16).numpy()
        else:
            data = self.data.astype(dtype.to_numpy())
        return WeightData(
            data=data, name=self.name, dtype=dtype, shape=Shape(data.shape)
        )

    def view(self, dtype: DType) -> WeightData:
        if self.dtype == dtype:
            return self

        # Compute the new shape for the updated dtype.
        if dtype == DType.bfloat16:
            assert torch is not None
            data = torch.from_numpy(self.data).view(dtype)
        else:
            data = self.data.view(dtype.to_numpy())
        return dataclasses.replace(self, dtype=dtype, shape=Shape(data.shape))

    def __repr__(self):
        return f"WeightData({self.dtype}, {self.shape})"


WeightsAdapter = Callable[..., dict[str, WeightData]]
