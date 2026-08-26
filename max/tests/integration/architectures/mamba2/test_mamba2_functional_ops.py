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
"""Graph-construction smoke test for the Mamba2 SSD functional-op wrapper.

This verifies that ``ssd_chunk_scan_combined`` from
``max.pipelines.architectures.mamba2.functional_ops`` resolves through
``ops.custom`` to the registered ``ssd_chunk_scan_combined`` Mojo kernel,
compiles inside a ``Graph``, executes end-to-end, and returns a tensor
of the expected shape with all finite values.

The current Mojo kernel
(``max/kernels/src/state_space/ssd_chunk_combined_ops.mojo``) only wires
the CPU dispatch — the GPU branch raises explicitly until later RFC 0002
phases. The smoke test therefore runs on CPU regardless of the host's
accelerator count. Numerical parity vs the torch reference is intentionally
NOT asserted here; that is covered by the model-level parity gate in
RFC 0003 item 7.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.mamba2.functional_ops import (
    ssd_chunk_scan_combined,
)


def test_ssd_chunk_scan_combined_smoke() -> None:
    """Build, compile, and run a minimal SSD chunk-scan graph on CPU.

    Shapes are intentionally small but obey the kernel's invariants:

    * ``seqlen`` (4) is divisible by ``chunk_size`` (2);
    * ``B``/``C`` use ``ngroups == n_heads == 1``.
    """
    cpu = DeviceRef.CPU()
    batch = 1
    seqlen = 4
    n_heads = 1
    head_dim = 2
    state_dim = 3
    chunk_size = 2

    with Graph(
        "mamba2_ssd_chunk_scan_combined_smoke",
        input_types=[
            TensorType(
                DType.float32, [batch, seqlen, n_heads, head_dim], device=cpu
            ),
            TensorType(DType.float32, [batch, seqlen, n_heads], device=cpu),
            TensorType(DType.float32, [n_heads], device=cpu),
            TensorType(
                DType.float32,
                [batch, seqlen, n_heads, state_dim],
                device=cpu,
            ),
            TensorType(
                DType.float32,
                [batch, seqlen, n_heads, state_dim],
                device=cpu,
            ),
        ],
    ) as graph:
        x_v = graph.inputs[0].tensor
        dt_v = graph.inputs[1].tensor
        A_v = graph.inputs[2].tensor
        B_v = graph.inputs[3].tensor
        C_v = graph.inputs[4].tensor

        y_v, final_state_v = ssd_chunk_scan_combined(
            x_v, dt_v, A_v, B_v, C_v, chunk_size=chunk_size
        )
        graph.output(y_v, final_state_v)

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    cpu_device = model.input_devices[0]

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch, seqlen, n_heads, head_dim)).astype(
        np.float32
    )
    # Small positive dt keeps the discretized decay well-behaved.
    dt_np = (
        np.abs(rng.standard_normal((batch, seqlen, n_heads))) * 0.1
    ).astype(np.float32)
    # Negative A produces a contracting state recurrence (Mamba2 convention).
    A_np = -np.abs(rng.standard_normal((n_heads,))).astype(np.float32)
    B_np = rng.standard_normal((batch, seqlen, n_heads, state_dim)).astype(
        np.float32
    )
    C_np = rng.standard_normal((batch, seqlen, n_heads, state_dim)).astype(
        np.float32
    )

    outs = model.execute(
        md.Buffer.from_numpy(x_np).to(cpu_device),
        md.Buffer.from_numpy(dt_np).to(cpu_device),
        md.Buffer.from_numpy(A_np).to(cpu_device),
        md.Buffer.from_numpy(B_np).to(cpu_device),
        md.Buffer.from_numpy(C_np).to(cpu_device),
    )

    # Outputs are on CPU, so Buffer.to_numpy() round-trips without dlpack.
    y = outs[0].to_numpy()
    assert y.shape == (batch, seqlen, n_heads, head_dim)
    assert np.all(np.isfinite(y)), (
        "ssd_chunk_scan_combined Y output has non-finite values"
    )

    final_state = outs[1].to_numpy()
    assert final_state.shape == (batch, n_heads, head_dim, state_dim)
    assert np.all(np.isfinite(final_state)), (
        "ssd_chunk_scan_combined final_state output has non-finite values"
    )
