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
        # Unsupported types are commented out.
        _GGML_TO_DTYPE = {
            # torch.bool: DType.bool,
            gguf.GGMLQuantizationType.I8: DType.int8,
            gguf.GGMLQuantizationType.I16: DType.int16,
            gguf.GGMLQuantizationType.I32: DType.int32,
            gguf.GGMLQuantizationType.I64: DType.int64,
            #: DType.uint8,
            #: DType.uint16,
            #: DType.uint32,
            #: DType.uint64,
            gguf.GGMLQuantizationType.F16: DType.float16,
            gguf.GGMLQuantizationType.F32: DType.float32,
            gguf.GGMLQuantizationType.F64: DType.float64,
            gguf.GGMLQuantizationType.BF16: DType.bfloat16,
        }


def load_gguf(filepath) -> Dict[str, Weight]:
    _check_gguf()
    graph = Graph.current

    reader = gguf.GGUFReader(filepath)
    ret = {}
    for tensor in reader.tensors:
        dtype = _GGML_TO_DTYPE.get(tensor.tensor_type)
        ret[tensor.name] = graph.add_weight(
            tensor.name,
            dtype,
            # Dims are reversed for some reason:
            # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
            # We have to un-reverse them here.
            reversed(tensor.shape.tolist()),
            filepath,
            tensor.data_offset,
        )
    return ret
