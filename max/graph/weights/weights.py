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
from max._core_types.driver import DLPackArray
from max.driver import CPU, Tensor
from max.dtype import DType

from .. import Graph, TensorType
from ..quantization import QuantizationEncoding
from ..type import DeviceRef, Shape, ShapeLike
from ..weight import Weight

_Self = TypeVar("_Self", bound="Weights")

_INF_SESSION = None
_CAST_MODEL = None


def _cast_to_dtype(
    raw_tensor: DLPackArray, old_dtype: DType, new_dtype: DType
) -> Tensor:
    # FIXME: This is a circular dep
    from max.engine import InferenceSession  # type: ignore

    tensor = Tensor.from_dlpack(raw_tensor)

    original_shape = tensor.shape
    global _INF_SESSION
    if not _INF_SESSION:
        _INF_SESSION = InferenceSession(devices=[CPU()])

    global _CAST_MODEL
    if not _CAST_MODEL:
        with Graph(
            "cast",
            input_types=[
                TensorType(
                    dtype=old_dtype,
                    shape=["dim"],
                    device=DeviceRef.from_device(CPU()),
                )
            ],
        ) as graph:
            graph.output(graph.inputs[0].cast(new_dtype))  # type: ignore

        _CAST_MODEL = _INF_SESSION.load(graph)

    result = _CAST_MODEL(tensor.view(old_dtype, [tensor.num_elements]))[0]
    assert isinstance(result, Tensor)
    return result.view(new_dtype, original_shape)


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

    def __getattr__(self: _Self, attr) -> _Self: ...  # noqa: ANN001

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
        device: DeviceRef = DeviceRef.CPU(),
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

    data: DLPackArray
    name: str
    dtype: DType
    shape: Shape
    quantization_encoding: Optional[QuantizationEncoding] = None

    def __dlpack__(self) -> Any:
        return self.data.__dlpack__()

    def __dlpack_device__(self) -> Any:
        return self.data.__dlpack_device__()

    @classmethod
    def from_numpy(cls, arr, name):  # noqa: ANN001
        return cls(arr, name, DType.from_numpy(arr.dtype), Shape(arr.shape))

    def astype(self, dtype: DType) -> WeightData:
        if self.dtype == dtype:
            return self
        data = _cast_to_dtype(self.data, self.dtype, dtype)
        return WeightData(
            data=data,
            name=self.name,
            dtype=dtype,
            shape=Shape(data.shape),
        )

    def __repr__(self) -> str:
        return f"WeightData({self.dtype}, {self.shape})"


WeightsAdapter = Callable[..., dict[str, WeightData]]
