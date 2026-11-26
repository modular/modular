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
"""Runs the Engram static initializer into fp8 host buffers.

The graph compiler accepts FP8 host data movement only in a session with an
accelerator, which is how the static initializer runs in deployment.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.pipelines.architectures.deepseekV41_modulev3.layers import (
    engram_init,
)
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    unpack_e2m1_to_e4m3,
)

# Not a multiple of the rows per slice the test sets.
ROWS = 37
DIM = 256


@pytest.mark.parametrize(
    ("part", "packed"),
    [("weight", False), ("weight", True), ("scale", False)],
)
def test_static_initializer_writes_the_fp8_codes(
    part: str, packed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pinned table holds the checkpoint's codes, in its own dtype."""
    monkeypatch.setattr(engram_init, "_CHUNK_ROWS", 5)
    dtype, width = (
        (DType.float8_e4m3fn, DIM)
        if part == "weight"
        else (DType.float8_e8m0fnu, DIM // 32)
    )
    source_dtype, source_width = (
        (DType.uint8, width // 2) if packed else (dtype, width)
    )
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (ROWS, source_width), dtype=np.uint8)
    # Every E2M1 code, low and high nibble.
    if packed:
        source[0] = np.arange(source_width) % 256
    cpu = DeviceRef.CPU()
    with Graph(
        "engram_static_init",
        input_types=[
            BufferType(dtype, [ROWS, width], cpu),
            TensorType(source_dtype, [ROWS, source_width], cpu),
        ],
    ) as graph:
        destination, table = graph.inputs
        engram_init._initialize_table(
            destination.buffer, table.tensor, packed=packed
        )
        graph.output()
    model = InferenceSession(devices=[CPU(), Accelerator()]).load(graph)
    out = Buffer.from_numpy(np.zeros((ROWS, width), np.uint8))
    model.execute(out.view(dtype), Buffer.from_numpy(source).view(source_dtype))
    expected = np.empty((ROWS, width), np.uint8)
    if packed:
        unpack_e2m1_to_e4m3(source, expected)
    else:
        expected[:] = source
    np.testing.assert_array_equal(out.to_numpy(), expected)
