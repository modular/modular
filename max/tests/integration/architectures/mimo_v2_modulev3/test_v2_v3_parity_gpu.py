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

The V3 port caches both attention groups at their native 192/128 head dims
while V2 pads them to 256, so the two run different attention kernels and
their logits differ by more than the residual fusion alone. The gate is
therefore each port against the HF float32 goldens: V3 (native) must be no
worse than V2 (padded), under each of the runner's ``SCHEDULES``: chunked
prefill and decode, prefill chunks longer than the sliding window, and
8-token verify steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tiny_fixture
from max.driver import accelerator_count
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MHAAttnKey, MHAKVCacheParams
from mimo_runner import Runner as GraphRunner
from mimo_v3_runner import SCHEDULES, kv_params
from mimo_v3_runner import Runner as ModuleV3Runner
from transformers.configuration_utils import PretrainedConfig

GOLDENS = (
    Path(__file__).parent.parent / "mimo_v2" / "testdata" / "tiny_goldens.npz"
)
"""HF float32 logits of the fixture (``test_serving_numerics_gpu.py``)."""

MEAN_RATIO, MAX_KL, TOP1_SLACK = 1.25, 2.5e-2, 0.01
"""V3's mean KL(HF || V3) over every row of both prompts may be at most
``MEAN_RATIO`` times V2's, its worst row at most ``MAX_KL``, and its top-1
agreement with HF at most ``TOP1_SLACK`` below V2's.

``MEAN_RATIO`` is the design's numerics bound for the native kernels (error
against float32 at most 1.25x the padded path's). ``MAX_KL`` is the HF
numerics test's bound: a single row's KL is dominated by near-ties, so the
worst row is held to the defect bound rather than a ratio. TP2 B200 actuals
(mean / max / top-1 against HF): chunked and long_chunk V2 1.09e-3 /
6.60e-3 / 0.972, V3 1.08e-3 / 6.50e-3 / 0.972; verify V2 1.03e-3 / 5.40e-3
/ 0.972, V3 1.09e-3 / 6.50e-3 / 0.976. V3 against V2 directly: mean 1.2e-4
(chunked) and 3.6e-4 (verify), a tenth of either's distance from HF; with
the padded cache (the same kernels in both ports) it was 2.4e-5.

A value written to the wrong columns of the split cache moves V3's mean
KL against HF by orders of magnitude, far past ``MEAN_RATIO``. Dropping the
sliding layers' sinks moves it to 2.5e-2, and widening their window by one
key moves the worst row to 0.35."""


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


Logits = dict[str, dict[str, list[npt.NDArray[np.float32]]]]
"""Each port's logits per schedule, one array per prompt."""


@pytest.fixture(scope="module")
def logits_by_port(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Logits, list[tuple[tuple[int, ...], bool]]]:
    """Both ports' logits for the two prompts under each schedule,
    teacher-forced at TP2, and the V3 run's cache inputs."""
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
    results: Logits = {}
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
    results["v3"] = {
        name: [
            logits
            for logits in v3.teacher_forced(prompts, chunk, decode)
            if logits is not None
        ]
        for name, (chunk, decode) in SCHEDULES.items()
    }
    kv_blocks = v3.kv_blocks
    del v3
    v2 = GraphRunner(
        path,
        num_devices=num_devices,
        max_seq_len=256,
        kv_cache_bytes=256 * 1024**2,
    )
    # The V2 runner also returns each layer's hidden states, which the V3
    # port does not; the logits are the shared surface.
    results["v2"] = {
        name: [
            logits
            for logits, _hidden in v2.teacher_forced(prompts, chunk, decode)
            if logits is not None
        ]
        for name, (chunk, decode) in SCHEDULES.items()
    }
    del v2
    return results, kv_blocks


def test_v3_caches_both_groups_unpadded(
    logits_by_port: tuple[Logits, list[tuple[tuple[int, ...], bool]]],
) -> None:
    """The V3 runner's cache is the one the model config builds: each group
    in one 320-wide plane (192-wide keys, 128-wide values). Without this the
    gate below could compare padded against padded."""
    _, kv_blocks = logits_by_port
    num_devices = len(kv_blocks) // 2
    sliding_heads = tiny_fixture.SLIDING_KV_HEADS // num_devices
    full_heads = tiny_fixture.FULL_KV_HEADS // num_devices
    layers = len(tiny_fixture.LAYERS) // 2
    # Tree order: the sliding group's devices, then the full group's.
    assert [(shape[1:], split) for shape, split in kv_blocks] == [
        ((1, layers, 128, sliding_heads, 320), True)
    ] * num_devices + [
        ((1, layers, 128, full_heads, 320), True)
    ] * num_devices, kv_blocks


def test_v3_sliding_window_bounds_the_split_k_count() -> None:
    """The split cache's decode launches exactly the metadata's split-K
    count, which the sliding group's window holds at one partition however
    long the cache; the full group's grows with it."""
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    tree = kv_params(
        PretrainedConfig(**tiny_fixture.config()),
        [DeviceRef.GPU(i) for i in range(num_devices)],
        128,
    )
    counts = {}
    for name, group in tree.children.items():
        assert isinstance(group, MHAKVCacheParams)
        key = group.resolve_attn_key(1, 8, 262_144)
        assert isinstance(key, MHAAttnKey)
        counts[name] = key.num_partitions
    assert counts["sliding_attention"] == 1, counts
    assert counts["full_attention"] > 1, counts


@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_v3_numerics_no_worse_than_v2(
    logits_by_port: tuple[Logits, list[tuple[tuple[int, ...], bool]]],
    schedule: str,
) -> None:
    prompts = tiny_fixture.prompts()
    v2 = logits_by_port[0]["v2"][schedule]
    v3 = logits_by_port[0]["v3"][schedule]
    assert len(v2) == len(v3) == len(prompts)
    goldens = np.load(GOLDENS)
    vs_hf, top1 = {}, {}
    for port, arm in (("v2", v2), ("v3", v3)):
        golds = [goldens[f"logits_{name}"] for name in prompts]
        vs_hf[port] = np.concatenate(
            [_kl(gold, test) for gold, test in zip(golds, arm, strict=True)]
        )
        top1[port] = np.concatenate(
            [
                gold.argmax(-1) == test.argmax(-1)
                for gold, test in zip(golds, arm, strict=True)
            ]
        ).mean()
    direct = np.concatenate([_kl(a, b) for a, b in zip(v2, v3, strict=True)])
    summary = "; ".join(
        f"{port} vs HF KL mean {vs_hf[port].mean():.2e}, max "
        f"{vs_hf[port].max():.2e}, top-1 {top1[port]:.3f}"
        for port in ("v2", "v3")
    )
    summary = (
        f"{schedule}: {summary}; V2 vs V3 KL mean {direct.mean():.2e}, max "
        f"{direct.max():.2e}"
    )
    print(summary)
    assert vs_hf["v3"].mean() <= MEAN_RATIO * vs_hf["v2"].mean(), summary
    assert vs_hf["v3"].max() <= MAX_KL, summary
    assert top1["v3"] >= top1["v2"] - TOP1_SLACK, summary
