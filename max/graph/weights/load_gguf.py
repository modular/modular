# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for importing a GGUF checkpoint into a MAX Graph."""

from __future__ import annotations

import subprocess
import sys
from os import PathLike
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from max.graph import DeviceRef

# Only import gguf when used.
try:
    import gguf  # type: ignore
except ImportError:
    gguf = None


from max.dtype import DType

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from .weights import WeightData, Weights

_GGML_TO_DTYPE: dict[gguf.GGMLQuantizationType, DType] = {}
_FROM_QUANTIZED_GGML_DTYPES = {}
_TO_QUANTIZED_GGML_DTYPES = {}


def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def _check_gguf():
    global \
        gguf, \
        _GGML_TO_DTYPE, \
        _FROM_QUANTIZED_GGML_DTYPES, \
        _TO_QUANTIZED_GGML_DTYPES
    if gguf is None:
        _install("gguf")
        import gguf as _gguf

        gguf = _gguf

    if not _GGML_TO_DTYPE:
        _GGML_TO_DTYPE = {
            gguf.GGMLQuantizationType.I8: DType.int8,
            gguf.GGMLQuantizationType.I16: DType.int16,
            gguf.GGMLQuantizationType.I32: DType.int32,
            gguf.GGMLQuantizationType.I64: DType.int64,
            gguf.GGMLQuantizationType.F16: DType.float16,
            gguf.GGMLQuantizationType.F32: DType.float32,
            gguf.GGMLQuantizationType.F64: DType.float64,
            gguf.GGMLQuantizationType.BF16: DType.bfloat16,
        }

        _FROM_QUANTIZED_GGML_DTYPES = {
            gguf.GGMLQuantizationType.Q4_0: QuantizationEncoding.Q4_0,
            # gguf.GGMLQuantizationType.Q4_1,
            # gguf.GGMLQuantizationType.Q5_0,
            # gguf.GGMLQuantizationType.Q5_1,
            # gguf.GGMLQuantizationType.Q8_0,
            # gguf.GGMLQuantizationType.Q8_1,
            # gguf.GGMLQuantizationType.Q2_K,
            # gguf.GGMLQuantizationType.Q3_K,
            gguf.GGMLQuantizationType.Q4_K: QuantizationEncoding.Q4_K,
            gguf.GGMLQuantizationType.Q5_K: QuantizationEncoding.Q5_K,
            gguf.GGMLQuantizationType.Q6_K: QuantizationEncoding.Q6_K,
            # gguf.GGMLQuantizationType.Q8_K,
            # gguf.GGMLQuantizationType.IQ2_XXS,
            # gguf.GGMLQuantizationType.IQ2_XS,
            # gguf.GGMLQuantizationType.IQ3_XXS,
            # gguf.GGMLQuantizationType.IQ1_S,
            # gguf.GGMLQuantizationType.IQ4_NL,
            # gguf.GGMLQuantizationType.IQ3_S,
            # gguf.GGMLQuantizationType.IQ2_S,
            # gguf.GGMLQuantizationType.IQ4_XS,
            # gguf.GGMLQuantizationType.IQ1_M,
        }
        _TO_QUANTIZED_GGML_DTYPES = {
            value: key for key, value in _FROM_QUANTIZED_GGML_DTYPES.items()
        }


class GGUFWeights(Weights):
    _reader: gguf.GGUFReader
    _tensors: dict[str, gguf.ReaderTensor]
    _prefix: str
    _allocated: dict[str, np.ndarray]

    def __init__(
        self,
        source: Union[PathLike, gguf.GGUFReader],
        tensors=None,
        prefix: str = "",
        allocated=None,
    ):
        """Creates a GGUF weights reader.

        Args:
            source: Path to a GGUF file or a GGUFReader object.
            tensors: List of tensors in the GGUF checkpoint.
            prefix: Weight name or prefix.
            allocated: Dictionary of allocated values.
        """
        _check_gguf()
        assert gguf is not None
        from ._gguf_reader import TokenSkippingGGUFReader

        self._reader = (
            source
            if isinstance(source, gguf.GGUFReader)
            else TokenSkippingGGUFReader(source)
        )
        self._tensors = tensors or {t.name: t for t in self._reader.tensors}
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    def items(self):
        """Iterate through all allocable weights that start with the prefix."""
        for name in self._tensors:
            if name.startswith(self._prefix):
                yield (
                    name,
                    GGUFWeights(
                        self._reader,
                        self._tensors,
                        prefix=name,
                        allocated=self._allocated,
                    ),
                )

    def __getattr__(self, attr) -> GGUFWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        return GGUFWeights(
            self._reader,
            self._tensors,
            prefix=full_path,
            allocated=self._allocated,
        )

    def __getitem__(self, idx: int | str) -> GGUFWeights:
        return self.__getattr__(str(idx))

    def raw_tensor(self) -> npt.NDArray[Any]:
        """Returns the numpy tensor corresponding to this weights object.

        Raises:
            KeyError if this weights object isn't a tensor.
        """
        return self._raw_tensor().data

    def data(self) -> WeightData:
        tensor = self._raw_tensor()

        try:
            dtype = _GGML_TO_DTYPE[tensor.tensor_type]
        except KeyError as e:
            if tensor.tensor_type in _FROM_QUANTIZED_GGML_DTYPES:
                dtype = DType.uint8
            else:
                raise e

        # Dims are reversed for some reason:
        # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
        # We have to un-reverse them here.
        shape_list = list(reversed(tensor.shape.tolist()))
        if tensor.tensor_type in _FROM_QUANTIZED_GGML_DTYPES:
            shape = Shape(
                gguf.quant_shape_to_byte_shape(shape_list, tensor.tensor_type)
            )
        else:
            shape = Shape(shape_list)
        quantization_encoding = _FROM_QUANTIZED_GGML_DTYPES.get(
            tensor.tensor_type
        )
        return WeightData(
            tensor.data, self.name, dtype, shape, quantization_encoding
        )

    def _raw_tensor(self) -> gguf.ReaderTensor:
        """Returns the GGUF tensor corresponding to this weights object.

        Raises:
            KeyError if this weights object isn't a tensor.
        """
        if self._prefix not in self._tensors:
            raise KeyError(
                f"Could not find weight named {self._prefix}. Please check that"
                " the name is correct."
            )

        return self._tensors[self._prefix]

    def exists(self) -> bool:
        return self._prefix in self._tensors

    def _parse_weight(
        self, tensor: gguf.ReaderTensor, device: DeviceRef
    ) -> Weight:
        # Dims are reversed for some reason:
        # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
        # We have to un-reverse them here.
        shape = list(reversed(tensor.shape.tolist()))
        encoding = None
        if (dtype := _GGML_TO_DTYPE.get(tensor.tensor_type)) is None:
            if encoding := _FROM_QUANTIZED_GGML_DTYPES.get(tensor.tensor_type):
                # Quantized dtypes are treated as uint8 values.
                dtype = DType.uint8
                shape = gguf.quant_shape_to_byte_shape(
                    shape, tensor.tensor_type
                )
            else:
                raise ValueError(f"Unknown GGML DType: {tensor.tensor_type}.")

        return Weight(
            name=tensor.name,
            dtype=dtype,
            shape=shape,
            quantization_encoding=encoding,
            align=self._reader.alignment,
            device=device,
        )

    def allocate(
        self,
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
        device=DeviceRef.CPU(),
    ) -> Weight:
        """Creates and optionally validates a new Weight."""
        tensor = self._raw_tensor()
        weight = self._parse_weight(tensor, device)
        self._allocated[self._prefix] = tensor.data

        # Validate the loaded weight.
        if shape is not None:
            expected_shape = tuple(shape)
            weight_unpacked_shape = tuple(int(dim) for dim in weight.shape)
            if weight.quantization_encoding:
                # Get the unpacked weight.
                ggml_dtype = _TO_QUANTIZED_GGML_DTYPES[
                    weight.quantization_encoding
                ]
                weight_unpacked_shape = gguf.quant_shape_from_byte_shape(
                    weight_unpacked_shape, ggml_dtype
                )

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

        # GGUF checkpoints can be mixed precision, so we don't check if the
        # quantization encoding matches exactly.
        if quantization_encoding and not weight.quantization_encoding:
            raise ValueError(
                "Expected quantized weight but checkpoint contained"
                f" unquantized weight for {self._prefix}"
            )

        return weight

    @property
    def allocated_weights(self) -> dict[str, npt.NDArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
