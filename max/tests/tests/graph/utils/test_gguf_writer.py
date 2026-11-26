# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Tests for writing GGUF checkpoints with the ``GGUFWriter`` binding."""

from pathlib import Path

import numpy as np
import pytest
from max._core.gguf import GGUFWriter, gguf_open
from max.driver import Buffer
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import GGUFWeights

# GGML tensor type ids, as recorded in the GGUF header.
_GGML_F32 = 0
_GGML_BF16 = 30
_GGML_I32 = 26
_GGML_Q4_K = 12


def _raw_bytes(data) -> bytes:  # noqa: ANN001
    """The tensor's stored bytes, whatever its dtype (numpy lacks bfloat16)."""
    buffer: Buffer = data.data
    if len(data.shape) == 0:
        # A rank-0 buffer cannot be viewed as a different-width dtype.
        buffer = buffer.view(data.dtype, [1])
    return np.from_dlpack(buffer.view(DType.uint8)).tobytes()


def test_write_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    floats = np.arange(8, dtype=np.float32).reshape(2, 4)
    ints = np.array([1, -2, 3], dtype=np.int32)
    # Two Q4_K blocks: 256 elements each, stored as 144 bytes each.
    blocks = np.zeros((2, 144), dtype=np.uint8)

    writer = GGUFWriter()
    writer.set_string("general.architecture", "example")
    writer.set_uint("example.block_count", 4)
    writer.set_int("example.head_count", -1)
    writer.set_float("example.eps", 1e-5)
    writer.set_bool("example.has_bias", True)
    writer.add_tensor("a", _GGML_F32, [2, 4], floats.tobytes())
    writer.add_tensor("b", _GGML_I32, [3], ints.tobytes())
    writer.add_tensor("quantized", _GGML_Q4_K, [2, 256], blocks.tobytes())
    writer.write(path)

    weights = GGUFWeights(path)

    a = weights.a.data()
    assert a.dtype == DType.float32
    assert list(a.shape) == [2, 4]
    np.testing.assert_array_equal(np.from_dlpack(a.data), floats)

    b = weights.b.data()
    assert b.dtype == DType.int32
    np.testing.assert_array_equal(np.from_dlpack(b.data), ints)

    # Quantized tensors read back as uint8 blocks with the last dimension
    # scaled from elements to blocks.
    quantized = weights.quantized.data()
    assert quantized.dtype == DType.uint8
    assert list(quantized.shape) == [2, 144]
    assert quantized.quantization_encoding == QuantizationEncoding.Q4_K


def test_written_weight_is_allocatable(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    bf16 = np.ones((2,), dtype=np.uint16)
    writer = GGUFWriter(alignment=64)
    writer.add_tensor("w", _GGML_BF16, [2], bf16.tobytes())
    writer.write(path)

    weights = GGUFWeights(path)
    weight = weights.w.allocate(dtype=DType.bfloat16, shape=[2])
    assert weight.dtype == DType.bfloat16
    assert weight.align == 64
    with pytest.raises(ValueError, match="different shape"):
        weights.w.allocate(shape=[4])


def test_round_trips_committed_fixture(
    tmp_path: Path, testdata_directory: Path
) -> None:
    """A checkpoint written from the fixture's tensors reads back identically."""
    source_path = testdata_directory / "example_data.gguf"
    reader = gguf_open(source_path)
    source = GGUFWeights(source_path)
    entries = dict(source.items())

    path = tmp_path / "copy.gguf"
    writer = GGUFWriter()
    for name, weight in entries.items():
        data = weight.data()
        writer.add_tensor(
            name,
            {
                DType.float32: _GGML_F32,
                DType.int32: _GGML_I32,
                DType.int64: 27,
                DType.float64: 28,
                DType.bfloat16: _GGML_BF16,
                DType.uint8: _GGML_Q4_K,
            }[data.dtype],
            # Quantized tensors report a byte shape; the writer wants elements.
            list(reader.element_shape(name)),
            _raw_bytes(data),
        )
    writer.write(path)

    copied = dict(GGUFWeights(path).items())
    assert set(copied) == set(entries)
    for name, weight in copied.items():
        original = entries[name].data()
        data = weight.data()
        assert data.dtype == original.dtype, name
        assert list(data.shape) == list(original.shape), name
        assert _raw_bytes(data) == _raw_bytes(original), name


def test_rejects_mismatched_tensor_bytes() -> None:
    writer = GGUFWriter()
    # Four float32 elements are 16 bytes, not 15.
    with pytest.raises(ValueError, match="bytes"):
        writer.add_tensor("a", _GGML_F32, [4], b"\0" * 15)


def test_rejects_bad_alignment() -> None:
    with pytest.raises(ValueError, match="power of two"):
        GGUFWriter(alignment=48)
