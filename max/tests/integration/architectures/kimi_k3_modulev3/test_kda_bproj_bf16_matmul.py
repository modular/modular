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
"""The KDA beta-logits matmul in bf16 with an fp32 output.

`kda_attention` computes the beta logits as a bf16 x bf16 matmul with
`out_dtype=float32` (the weight is bf16 on device; bf16xbf16 products are
exact in the fp32 accumulator), which removes the per-layer elementwise
casts of `x` and the weight. This pins that path against the fp32 x fp32
matmul on the same bf16-rounded operands at the production per-rank
shape -- the beta logits feed the KDA gate, where a silent dtype
regression would not surface in the layer-output parity tolerance."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.compile_on_gpu("dispatch probe")
def test_bf16_beta_logits_match_fp32_path() -> None:
    rows, hidden, heads = 32, 7168, 12
    x_type = TensorType(DType.float32, [rows, hidden], device=DeviceRef.GPU())
    w_type = TensorType(DType.float32, [heads, hidden], device=DeviceRef.GPU())

    with Graph("probe", input_types=[x_type, w_type]) as graph:
        x, w = graph.inputs
        x_bf = ops.cast(x.tensor, DType.bfloat16)
        w_bf = ops.cast(w.tensor, DType.bfloat16)
        bf16_out = ops.matmul(
            x_bf, ops.transpose(w_bf, 0, 1), out_dtype=DType.float32
        )
        fp32_out = ops.matmul(
            ops.cast(x_bf, DType.float32),
            ops.transpose(ops.cast(w_bf, DType.float32), 0, 1),
        )
        graph.output(bf16_out, fp32_out)

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    summaries = model.kernel_summaries
    print("KERNEL SUMMARIES:")
    for s in summaries:
        print("  ", s)
    x_np = np.random.standard_normal((rows, hidden)).astype(np.float32)
    w_np = np.random.standard_normal((heads, hidden)).astype(np.float32)
    inputs = [
        Buffer.from_numpy(x_np).to(model.input_devices[0]),
        Buffer.from_numpy(w_np).to(model.input_devices[1]),
    ]
    bf16_result, fp32_result = model.execute(*inputs)
    bf16_np = bf16_result.to_numpy()
    fp32_np = fp32_result.to_numpy()
    rel = float(
        np.max(np.abs(bf16_np - fp32_np)) / max(np.max(np.abs(fp32_np)), 1e-12)
    )
    print("max rel diff bf16-path vs fp32-path:", rel)
    assert rel < 1e-4, rel
