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
"""GPU execution test for the V4.1 Engram layer-selection binding.

The hash-gather kernel itself (its production geometry, its pinned host
tables) is covered by ``//Kernels/test/engram:test_engram_ops`` and,
end to end, by the assembled ModuleV3 model; this package keeps the
execution coverage of the Engram selection code that rides the
gather's stacked output.
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.deepseekV41_modulev3.layers.engram import (
    _select_ordinal,
)

HEAD_DIM = 256
TOKENS = 9


def test_select_ordinal_executes_on_device() -> None:
    """``_select_ordinal`` runs and picks the call site's layer.

    The selection is Python code riding main's gather kernel output; it
    executes here over synthetic stacked operands whose rows encode
    ``(layer, token, column)``, so a wrong axis or index shows up
    immediately.
    """

    gpu = DeviceRef.GPU()
    stacked = np.zeros((2, TOKENS, 24 * HEAD_DIM), np.uint8)
    for layer in range(2):
        for t in range(TOKENS):
            for col in range(24):
                base = col * HEAD_DIM
                stacked[layer, t, base + 0] = layer
                stacked[layer, t, base + 1] = t
                stacked[layer, t, base + 2] = col
    with Graph(
        "dsv41_modulev3_engram_select",
        input_types=[
            TensorType(DType.uint8, [2, TOKENS, 24 * HEAD_DIM], gpu),
            TensorType(DType.uint32, [1], gpu),
        ],
    ) as graph:
        codes_v, ordinal_v = (value.tensor for value in graph.inputs)
        codes, _scales = _select_ordinal(codes_v, codes_v, ordinal_v)
        graph.output(codes)
    device = Accelerator()
    compiled = InferenceSession(devices=[device]).load(graph)

    got = (
        torch.from_dlpack(
            compiled.execute(
                torch.from_numpy(stacked).cuda(),
                torch.from_numpy(np.array([1], np.uint32)).cuda(),
            )[0]
        )
        .cpu()
        .view(torch.uint8)
        .numpy()
        .reshape(TOKENS, 24 * HEAD_DIM)
    )
    for t in range(TOKENS):
        for col in range(24):
            base = col * HEAD_DIM
            assert got[t, base + 0] == 1, (t, col)
            assert got[t, base + 1] == t, (t, col)
            assert got[t, base + 2] == col, (t, col)
