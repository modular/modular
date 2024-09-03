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
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# Only import gguf when used.
try:
    import gguf
except ImportError:
    gguf = None


from max.dtype import DType

from ..quantization import QuantizationEncoding
from ..type import ShapeLike
from ..weight import Weight

_GGML_TO_DTYPE = {}
_FROM_QUANTIZED_GGML_DTYPES = {}
_TO_QUANTIZED_GGML_DTYPES = {}


def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def _check_gguf():
    global gguf, _GGML_TO_DTYPE, _FROM_QUANTIZED_GGML_DTYPES, _TO_QUANTIZED_GGML_DTYPES
    if gguf is None:
        _install("sentencepiece")
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


def load_gguf(
    source: PathLike | gguf.GGUFReader,
) -> dict[str, tuple[Weight, npt.NDArray[Any]]]:
    _check_gguf()
    assert gguf is not None
    GGUFReader = gguf.GGUFReader

    reader = source if isinstance(source, GGUFReader) else GGUFReader(source)
    return {
        tensor.name: (parse_weight(tensor), tensor.data)
        for tensor in reader.tensors
    }


def parse_weight(tensor: gguf.ReaderTensor) -> Weight:
    _check_gguf()
    assert gguf is not None
    # Dims are reversed for some reason:
    # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
    # We have to un-reverse them here.
    shape = list(reversed(tensor.shape.tolist()))
    encoding = None
    if (dtype := _GGML_TO_DTYPE.get(tensor.tensor_type)) is None:
        if encoding := _FROM_QUANTIZED_GGML_DTYPES.get(tensor.tensor_type):
            # Quantized dtypes are treated as uint8 values.
            dtype = DType.uint8
            shape = gguf.quant_shape_to_byte_shape(shape, tensor.tensor_type)
        else:
            raise ValueError(f"Unknown GGML DType: {tensor.tensor_type}.")

    return Weight(
        name=tensor.name,
        dtype=dtype,
        shape=shape,
        quantization_encoding=encoding,
    )


class GGUFWeights:
    _reader: gguf.GGUFReader
    _tensors: dict[str, gguf.ReaderTensor]
    _prefix: str
    _allocated: dict[str, np.ndarray]

    def __init__(
        self,
        reader: gguf.GGUFReader,
        tensors=None,
        prefix: str = "",
        allocated=None,
    ):
        self._reader = reader
        self._tensors = tensors or {t.name: t for t in reader.tensors}
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated

    def __iter__(self):
        for name, tensor in self._tensors.items():
            if name.startswith(self._prefix):
                yield tensor

    def __getattr__(self, attr) -> GGUFWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        if not any(name.startswith(full_path) for name in self._tensors):
            raise AttributeError(f"No weight {full_path} found")
        return GGUFWeights(
            self._reader,
            self._tensors,
            prefix=full_path,
            allocated=self._allocated,
        )

    def __getitem__(self, idx: int) -> GGUFWeights:
        return self.__getattr__(str(idx))

    def tensor(self) -> gguf.ReaderTensor:
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

    def allocate(
        self,
        dtype: DType,
        shape: ShapeLike,
        quantization_encoding: Optional[QuantizationEncoding] = None,
    ) -> Weight:
        tensor = self.tensor()
        weight = parse_weight(tensor)
        self._allocated[self._prefix] = tensor.data

        # Validate the loaded weight.
        expected_shape = tuple(shape)
        weight_unpacked_shape = tuple(dim for dim in weight.shape)
        if weight.quantization_encoding:
            # Get the unpacked weight.
            ggml_dtype = _TO_QUANTIZED_GGML_DTYPES[weight.quantization_encoding]
            weight_unpacked_shape = gguf.quant_shape_from_byte_shape(
                weight_unpacked_shape, ggml_dtype
            )

        if (weight.dtype, weight_unpacked_shape) != (
            dtype,
            expected_shape,
        ):
            raise ValueError(
                "Did not get expected weight shape and/or dtype.\n\tExpected"
                f" dtype: {dtype}, got: {weight.dtype}\n\tExpected shape:"
                f" {expected_shape}, got: {weight_unpacked_shape}"
            )

        # GGUF checkpoints can be mixed precision, so we don't check if the
        # quantization encoding matches exactly.
        if quantization_encoding and not weight.quantization_encoding:
            raise ValueError(
                "Expected quantized weight but checkpoint contained"
                f" unquantized weight for {self._prefix}"
            )
        if weight.quantization_encoding and not quantization_encoding:
            raise ValueError("Expected quantized weight")

        return weight

    @property
    def allocated_weights(self) -> dict[str, np.ndarray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
