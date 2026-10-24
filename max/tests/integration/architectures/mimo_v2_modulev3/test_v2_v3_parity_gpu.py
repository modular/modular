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
"""The ModuleV3 migration's parity gate: graph-API MiMo-V2 vs ModuleV3 port.

Both ports run the same tiny fixture checkpoint teacher-forced over the
same paged KV cache discipline, and their logits are compared row by row.
The checkpoint, adapter renames, quantization and attention math are shared,
so a disagreement localizes to what the migration changed: the ModuleV3
graph (residual reduction through ``Tensor.to(mapping)`` fused with the
RMSNorm — ``multiply_before_cast`` where the V2 graph scales in BF16 after
the residual add — one shared MoE subgraph, broadcast collectives instead of
signals).

The V3 arch returns no hidden states, so parity is gated on logits — the
same surface the V2 test scores against HF, and the one both ports share
exactly.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import pytest
import tiny_fixture
from max.driver import accelerator_count
from mimo_runner import Runner as GraphRunner
from mimo_v3_runner import Runner as ModuleV3Runner

CHUNK, DECODE = 96, 4

MEAN_KL, MAX_KL, TOP1 = 1e-4, 1e-3, 0.99
"""Bounds on KL(V2 || V3) per row, and on top-1 agreement, over every row of
both prompts.

Both arms share the quantized weights, the reference attention and the
same tolerance discipline, so what remains is kernel-order noise (the fused
residual+norm above all). First TP2 B200 run measured mean 2.44e-05, max
1.18e-04, top-1 1.000; the bounds sit an order of magnitude above those
actuals. The random logits are flat
(a mean top probability of 0.12), so near-ties cost top-1 at the same rate
as in the HF comparison. The first box run records the actuals here; if they
sit far below these bounds, tighten rather than leave slack a defect could
hide in."""


def _kl(
    gold: npt.NDArray[np.float32], test: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Each row's KL(gold || test)."""
    lg, lt = (
        x.astype(np.float64)
        - np.logaddexp.reduce(x.astype(np.float64), -1, keepdims=True)
        for x in (gold, test)
    )
    return (np.exp(lg) * (lg - lt)).sum(-1)


@pytest.fixture(scope="module")
def logits_by_port(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, list[npt.NDArray[np.float32]]]:
    """Both ports' logits for the two prompts, teacher-forced at TP2."""
    checkpoint = tiny_fixture.tensors()
    path = tmp_path_factory.mktemp("mimo_parity")
    tiny_fixture.write(path, checkpoint)
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    assert accelerator_count() >= num_devices
    prompts = [
        np.array(ids, np.int64) for ids in tiny_fixture.prompts().values()
    ]
    results: dict[str, list[npt.NDArray[np.float32]]] = {}
    # The V3 runner builds first: a precompile record run stops at the
    # first hardware need (V2's Signals.allocate, V3's KV pool), so this
    # order is what lets it record the V3 graph — the one under test — and
    # leave the known-good V2 compile to the GPU's past-stop path.
    v3 = ModuleV3Runner(
        path,
        num_devices=num_devices,
        max_seq_len=256,
        kv_cache_bytes=256 * 1024**2,
    )
    results["v3"] = [
        logits
        for logits in v3.teacher_forced(prompts, CHUNK, DECODE)
        if logits is not None
    ]
    del v3
    v2 = GraphRunner(
        path,
        num_devices=num_devices,
        max_seq_len=256,
        kv_cache_bytes=256 * 1024**2,
    )
    # The V2 runner also returns each layer's hidden states, which the V3
    # port does not; the logits are the shared surface.
    results["v2"] = [
        logits
        for logits, _hidden in v2.teacher_forced(prompts, CHUNK, DECODE)
        if logits is not None
    ]
    del v2
    return results


def test_v3_logits_match_v2(
    logits_by_port: dict[str, list[npt.NDArray[np.float32]]],
) -> None:
    prompts = tiny_fixture.prompts()
    v2, v3 = logits_by_port["v2"], logits_by_port["v3"]
    assert len(v2) == len(v3) == len(prompts)
    kl, top1 = {}, {}
    for name, gold, test in zip(prompts, v2, v3, strict=True):
        kl[name] = _kl(gold, test)
        top1[name] = gold.argmax(-1) == test.argmax(-1)
    rows = np.concatenate(list(kl.values()))
    agree = np.concatenate(list(top1.values())).mean()
    edge = tiny_fixture.BEACON_ROW + tiny_fixture.SLIDING_WINDOW
    summary = (
        f"parity KL mean {rows.mean():.2e}, max {rows.max():.2e} (row {edge} "
        f"of the long prompt {kl['long'][edge]:.2e}); top-1 {agree:.3f}"
    )
    print(summary)
    assert rows.mean() <= MEAN_KL, summary
    assert rows.max() <= MAX_KL, summary
    assert agree >= TOP1, summary
