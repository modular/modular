# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, TypeVar

import numpy.typing as npt
from max.dtype import DType

from ..quantization import QuantizationEncoding
from ..type import ShapeLike
from ..weight import Weight

_Self = TypeVar("_Self", bound="Weights")


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

        Raises:
            KeyError if this weights object isn't a tensor.
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


class WeightsConverter(Protocol):
    @staticmethod
    def load_weights(weight_path: list[Path], **kwargs) -> Weights:
        """Loads the converted weights from a path."""
        ...
