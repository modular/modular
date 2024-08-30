# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for importing a GGUF checkpoint into a MAX Graph."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from os import PathLike
from typing import Callable, Dict, Optional

# Only import gguf when used.
try:
    import gguf
except ImportError:
    gguf = None


from max.dtype import DType

from ..graph import Graph
from ..quantization import QuantizationEncoding
from ..value import Value
from ..weight import Weight

_GGML_TO_DTYPE = {}
_QUANTIZED_GGML_DTYPES = {}


def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def _check_gguf():
    global gguf, _GGML_TO_DTYPE, _QUANTIZED_GGML_DTYPES
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

        _QUANTIZED_GGML_DTYPES = {
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


def load_gguf(source: PathLike | gguf.GGUFReader) -> Dict[str, Weight]:
    _check_gguf()
    assert gguf is not None
    GGUFReader = gguf.GGUFReader

    reader = source if isinstance(source, GGUFReader) else GGUFReader(source)
    return {tensor.name: parse_weight(tensor) for tensor in reader.tensors}


def parse_weight(tensor: gguf.ReaderTensor) -> Weight:
    _check_gguf()
    assert gguf is not None
    # Dims are reversed for some reason:
    # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
    # We have to un-reverse them here.
    shape = list(reversed(tensor.shape.tolist()))
    encoding = None
    if (dtype := _GGML_TO_DTYPE.get(tensor.tensor_type)) is None:
        if encoding := _QUANTIZED_GGML_DTYPES.get(tensor.tensor_type):
            # Quantized dtypes are treated as uint8 values.
            dtype = DType.uint8
            shape = gguf.quant_shape_to_byte_shape(shape, tensor.tensor_type)
        else:
            raise ValueError(f"Unknown GGML DType: {tensor.tensor_type}.")

    return Weight(
        name=tensor.name,
        dtype=dtype,
        shape=shape,
        filepath=tensor.data.filename,
        offset=tensor.data_offset,
        quantization_encoding=encoding,
    )


class Weights:
    _reader: gguf.GGUFReader
    _tensors: dict[str, gguf.ReaderTensor]
    _prefix: str

    def __init__(self, reader: gguf.GGUFReader, tensors=None, prefix: str = ""):
        self._reader = reader
        self._tensors = tensors or {t.name: t for t in reader.tensors}
        self._prefix = prefix

    def __iter__(self):
        for name, tensor in self._tensors.items():
            if name.startswith(self._prefix):
                yield tensor

    def __getattr__(self, attr) -> Weight | Weights:
        full_path = f"{self._prefix}{attr}"
        if tensor := self._tensors.get(full_path):
            return parse_weight(tensor)
        elif not any(name.startswith(full_path) for name in self._tensors):
            raise AttributeError(f"No weight {full_path} found")
        return Weights(self._reader, self._tensors, prefix=full_path + ".")

    def __getitem__(self, idx: int) -> Weight | Weights:
        return self.__getattr__(str(idx))
