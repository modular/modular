# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for importing a GGUF checkpoint into a MAX Graph."""

import subprocess
import sys
from typing import Dict

# Only import gguf when used.
try:
    import gguf
except ImportError:
    gguf = None


from ..dtype import DType
from ..graph import Graph
from ..weight import Weight

_GGML_TO_DTYPE = {}


def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def _check_gguf():
    global gguf, _GGML_TO_DTYPE
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


def load_gguf(gguf_or_filepath) -> Dict[str, Weight]:
    _check_gguf()
    assert gguf is not None
    graph = Graph.current
    reader = gguf_or_filepath
    if not isinstance(reader, gguf.GGUFReader):
        reader = gguf.GGUFReader(gguf_or_filepath)
    ret = {}
    for tensor in reader.tensors:
        # Dims are reversed for some reason:
        # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
        # We have to un-reverse them here.
        shape = list(reversed(tensor.shape.tolist()))
        dtype = _GGML_TO_DTYPE.get(tensor.tensor_type, None)
        if dtype is None:
            # Quantized dtypes do not have a direct map to a graph DType and are
            # treated as uint8 values.
            dtype = DType.uint8
            shape = gguf.quant_shape_to_byte_shape(shape, tensor.tensor_type)
        ret[tensor.name] = graph.add_weight(
            tensor.name,
            dtype,
            shape,
            tensor.data.filename,
            tensor.data_offset,
        )
    return ret
