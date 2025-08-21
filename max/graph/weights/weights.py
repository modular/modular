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
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

import numpy.typing as npt
from max.driver import CPU, DLPackArray, Tensor
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
    """Protocol for managing and accessing model weights hierarchically.

    The Weights protocol provides a convenient interface for loading and organizing
    neural network weights. It supports hierarchical naming through attribute and
    index access, making it easy to work with complex model architectures.

    Weights in MAX are tensors backed by external memory (buffers or memory-mapped
    files) that remain separate from the compiled graph.

    .. code-block:: python

        from max.graph import Graph
        from max.dtype import DType

        # Create a graph and get its weights interface
        graph = Graph("my_model")
        weights = graph.weights()

        # Allocate weights with hierarchical naming
        attn_weight = weights.transformer.layers[0].attention.weight.allocate(
            dtype=DType.float32,
            shape=(768, 768)
        )
        # Creates weight named "transformer.layers.0.attention.weight"

        # Check if a weight exists before allocating
        if weights.transformer.layers[0].mlp.weight.exists():
            mlp_weight = weights.transformer.layers[0].mlp.weight.allocate(
                dtype=DType.float16,
                shape=(768, 3072)
            )
    """

    @property
    def name(self) -> str:
        """Get the current weight name or prefix.

        Returns:
            The hierarchical name built from attribute and index access.
            For example, if accessed as ``weights.model.layers[0]``,
            returns "model.layers.0".
        """
        ...

    def __getattr__(self: _Self, attr: str) -> _Self: ...

    def __getitem__(self: _Self, idx: int | str) -> _Self: ...

    def exists(self) -> bool:
        """Check if a weight with this exact name exists.

        .. code-block:: python

            if weights.model.classifier.weight.exists():
                classifier = weights.model.classifier.weight.allocate(...)
            else:
                print("Classifier weight not found")

        Returns:
            True if a weight with the current hierarchical name exists
            in the loaded weights, False otherwise.
        """
        ...

    def items(self: _Self) -> Iterator[tuple[str, _Self]]:
        """Iterate through all weights that start with the current prefix.

        .. code-block:: python

            # Iterate through all weights in a specific layer
            for name, weight in weights.transformer.layers[0].items():
                print(f"Found weight: {name}")

        Yields:
            Tuples of (name, weight_accessor) for each weight under the
            current prefix. The name is relative to the current prefix.
        """
        ...

    def data(self) -> WeightData:
        """Get weight data with metadata.

        .. code-block:: python

            weight_data = weights.model.embeddings.weight.data()
            print(f"Shape: {weight_data.shape}")
            print(f"Dtype: {weight_data.dtype}")

            # Convert to different dtype
            fp16_data = weight_data.astype(DType.float16)

        Returns:
            A WeightData object containing the tensor data along with
            metadata like name, dtype, shape, and quantization encoding.

        Raises:
            KeyError: If no weight exists at the current hierarchical name.
        """
        ...

    def allocate(
        self,
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> Weight:
        """Create a Weight object for this tensor.

        .. code-block:: python

            # Allocate a weight with specific configuration
            weight = weights.model.layers[0].weight.allocate(
                dtype=DType.float16,  # Convert to half precision
                shape=(768, 768),
                device=DeviceRef.GPU(0)  # Place on first GPU
            )

            # Add to graph
            with graph:
                weight_tensor = graph.add_weight(weight)

        Args:
            dtype: Data type for the weight. If ``None``, uses the original dtype.
            shape: Shape of the weight tensor. If ``None``, uses the original shape.
            quantization_encoding: Quantization scheme to apply (for example, ``Q4_K``, ``Q8_0``).
            device: Target device for the weight (CPU or GPU).

        Returns:
            A Weight object that can be added to a graph using
            ``graph.add_weight()``.
        """
        ...

    @property
    def allocated_weights(self) -> dict[str, DLPackArray]:
        """Get all previously allocated weights. This only includes weights that were explicitly allocated
            using the :meth:`allocate` method, not all available weights.

        Returns:
            A dictionary mapping weight names to their numpy arrays for
            all weights that have been allocated through this interface.
        """
        ...


@dataclasses.dataclass
class WeightData:
    """Container for weight tensor data with metadata.

    ``WeightData`` encapsulates a weight tensor along with its metadata,
    providing utilities for type conversion and format compatibility.
    It supports the DLPack protocol for efficient tensor sharing between
    frameworks.
    """

    data: DLPackArray
    """The weight tensor as a DLPack array."""
    name: str
    """Hierarchical name of the weight (for example, ``"model.layers.0.weight"``)."""

    dtype: DType
    """Data type of the tensor (for example, ``DType.float32``, ``DType.uint8``)."""

    shape: Shape
    """Shape of the tensor as a Shape object."""

    quantization_encoding: Optional[QuantizationEncoding] = None
    """Optional quantization scheme applied to the weight."""

    def __dlpack__(self) -> Any:
        return self.data.__dlpack__()

    def __dlpack_device__(self) -> Any:
        return self.data.__dlpack_device__()

    @classmethod
    def from_numpy(cls, arr: npt.NDArray[Any], name: str) -> WeightData:
        """Create WeightData from a numpy array.

        Args:
            arr: Numpy array containing the weight data.
            name: Name to assign to this weight.

        Returns:
            A new WeightData instance with dtype and shape inferred
            from the numpy array.
        """
        return cls(arr, name, DType.from_numpy(arr.dtype), Shape(arr.shape))

    def astype(self, dtype: DType) -> WeightData:
        """Convert the weight data to a different dtype.

        This method performs actual data conversion, unlike :meth:`view` which
        reinterprets the underlying bytes. Special handling is provided for
        bfloat16 conversions using PyTorch when available.

        .. code-block:: python

            # Convert float32 weights to float16 for reduced memory
            weight_data = weights.model.layer.weight.data()
            fp16_data = weight_data.astype(DType.float16)

        Args:
            dtype: Target data type for conversion.

        Returns:
            A new WeightData instance with the converted data.
        """
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
"""Type alias for functions that adapt weight formats to WeightData dictionaries.

WeightsAdapter functions are used by pipeline architectures to convert between
different checkpoint formats (e.g., HuggingFace, PyTorch) and MAX's internal
format. They take model configuration and return a dictionary mapping weight
names to WeightData objects.
"""
