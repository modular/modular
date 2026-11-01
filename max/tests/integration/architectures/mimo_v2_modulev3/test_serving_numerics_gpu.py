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
"""Scores MAX's ModuleV3 MiMo-V2 port's logits on the tiny fixture against HF
goldens.

The ModuleV3 counterpart of ``../mimo_v2/test_serving_numerics_gpu.py``. It
runs the same checkpoint and the same goldens (the fixture is arch-agnostic:
what differs is only the compiled graph the logits come from), through the
V3 runner: ``Module.compile`` over a ``DeviceMesh``, no signal buffers, no
per-layer hidden states, no device-graph capture (the arch declares
``supports_device_graph_capture`` false and the graph has no capture
timeline). Both prompts run as one ragged batch through the paged KV cache,
whose groups both hold keys and values at their native widths, under each of
the runner's ``SCHEDULES``: chunked prefill and decode, prefill chunks longer
than the sliding window, and 8-token verify steps.

Each fused chunked QKV mode (sliding, full, both) must produce exactly the
unfused model's logits under every schedule, so it meets the golden bounds
too: the fused op computes the same BF16 values as the unfused stores and
RoPE, so any difference is a fused-path defect the golden bounds would be too
loose to see.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tiny_fixture
from max.driver import accelerator_count
from mimo_v3_runner import SCHEDULES, Runner

GOLDENS = (
    Path(__file__).parent.parent / "mimo_v2" / "testdata" / "tiny_goldens.npz"
)
"""The V2 harness's goldens: HF float32 logits of the same tiny checkpoint,
computed by ``make_random_fixture.py --tiny-goldens``. Reused, not copied —
the fixture checkpoint and goldens are arch-agnostic."""

FUSED_MODES = [(True, False), (False, True), (True, True)]
"""(fused sliding, fused full) QKV preparation."""
SLIDING_LAYERS = sum(sliding for sliding, _ in tiny_fixture.LAYERS)
FULL_LAYERS = len(tiny_fixture.LAYERS) - SLIDING_LAYERS

MEAN_KL, MAX_KL, TOP1 = 4e-3, 2.5e-2, 0.92
"""Bounds on KL(HF || MAX) over the whole vocabulary, and on top-1
agreement, over every row of both prompts.

The same bounds as the V2 test (see its docstring for the measured TP2
actuals: mean KL 1.09e-3, maximum KL 6.6e-3, top-1 0.972, and the defects
each bound catches by 6x or more). The goldens decode the same quantized
weights in float32, so the differences from the V2 arm are the compiled
ModuleV3 graph and its cache: its residual reduces through
``Tensor.to(mapping)`` fused with the following RMSNorm
(``multiply_before_cast``, where the V2 graph scales in BF16 after the
residual add), its MoE layers share one compiled subgraph, and both
attention groups run the split cache's kernels at their native head dims.
The measured TP2 B200 actuals for this arm: mean KL 1.08e-3 (chunked and
long_chunk) and 1.09e-3 (verify), maximum KL 6.50e-3, top-1 0.972-0.976 —
every bound clears by 3x or more, so a touched graph shows up here before it
moves the V2-vs-V3 parity gate.

The porting bugs the no-GPU lane caught (missing ``o_proj``, the field-only
E8M0 retyping) do not survive the runner at all, so this gate guards the
graph's numerics, not the load contract; the CPU harness test pins the
latter, and ``test_v2_v3_parity_gpu.py`` compares against the graph-API
arch directly."""


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


Logits = dict[str, list[npt.NDArray[np.float32]]]
"""The logits per schedule, one array per prompt."""


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture's checkpoint, written to disk."""
    checkpoint = tiny_fixture.tensors()
    goldens = np.load(GOLDENS)
    assert str(goldens["digest"]) == tiny_fixture.digest(checkpoint), (
        "tiny_fixture.py no longer builds the checkpoint the goldens were "
        "made from; rerun make_random_fixture.py --tiny-goldens."
    )
    path = tmp_path_factory.mktemp("mimo_v3")
    tiny_fixture.write(path, checkpoint)
    return path


Run = tuple[Logits, list[tuple[tuple[int, ...], bool]]]
"""The logits per schedule and the cache inputs they ran on."""


def _logits(path: Path, *, sliding: bool, full: bool) -> Run:
    """The V3 model's logits for both prompts under each schedule,
    teacher-forced at TP2, and the cache inputs it ran on."""
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    assert accelerator_count() >= num_devices
    model = Runner(
        path,
        num_devices=num_devices,
        max_seq_len=256,
        kv_cache_bytes=256 * 1024**2,
        fused_sliding_qkv=sliding,
        fused_full_qkv=full,
    )
    # Attention runs inline, once per device per layer.
    assert model.fused_qkv_count == num_devices * (
        SLIDING_LAYERS * sliding + FULL_LAYERS * full
    )
    prompts = [
        np.array(ids, np.int64) for ids in tiny_fixture.prompts().values()
    ]
    outputs = {
        name: [
            output
            for output in model.teacher_forced(prompts, chunk, decode)
            if output is not None
        ]
        for name, (chunk, decode) in SCHEDULES.items()
    }
    return outputs, model.kv_blocks


# One fixture per compiled graph: the precompile record run stops each
# fixture at the hardware its runner needs right after compiling, so a fixture
# compiling several graphs would leave all but the first to the GPU worker.
@pytest.fixture(scope="module")
def logits(checkpoint_dir: Path) -> Run:
    """The unfused model's run."""
    return _logits(checkpoint_dir, sliding=False, full=False)


@pytest.fixture(
    scope="module", params=FUSED_MODES, ids=["sliding", "full", "both"]
)
def fused_logits(checkpoint_dir: Path, request: pytest.FixtureRequest) -> Run:
    """Each fused mode's run."""
    sliding, full = request.param
    return _logits(checkpoint_dir, sliding=sliding, full=full)


def test_both_groups_run_on_the_split_cache(logits: Run) -> None:
    """Each group's cache input is one 320-wide plane (192-wide keys,
    128-wide values), so the bounds below score the native kernels."""
    _, kv_blocks = logits
    num_devices = len(kv_blocks) // 2
    layers = len(tiny_fixture.LAYERS) // 2
    expected = [
        ((1, layers, 128, heads // num_devices, 320), True)
        for heads in (tiny_fixture.SLIDING_KV_HEADS, tiny_fixture.FULL_KV_HEADS)
        for _ in range(num_devices)
    ]
    assert [(shape[1:], split) for shape, split in kv_blocks] == expected


@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_serving_logits_match_hf(logits: Run, schedule: str) -> None:
    goldens = np.load(GOLDENS)
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    prompts = tiny_fixture.prompts()
    kl, top1 = {}, {}
    for name, output in zip(prompts, logits[0][schedule], strict=True):
        gold = goldens[f"logits_{name}"]
        kl[name] = _kl(gold, output)
        top1[name] = gold.argmax(-1) == output.argmax(-1)
    rows = np.concatenate(list(kl.values()))
    agree = np.concatenate(list(top1.values())).mean()
    edge = tiny_fixture.BEACON_ROW + tiny_fixture.SLIDING_WINDOW
    summary = (
        f"{schedule}, {num_devices} GPU(s): KL mean {rows.mean():.2e}, max "
        f"{rows.max():.2e} (row {edge} of the long prompt "
        f"{kl['long'][edge]:.2e}); top-1 {agree:.3f}"
    )
    print(summary)
    assert rows.mean() <= MEAN_KL, summary
    assert rows.max() <= MAX_KL, summary
    assert agree >= TOP1, summary


def test_fused_logits_match_unfused(
    # Set up first, so the record run reaches its compile before the unfused
    # fixture re-raises the hardware stop it recorded.
    fused_logits: Run,
    logits: Run,
) -> None:
    """Fused preparation writes the same split cache the unfused stores do,
    so every schedule's logits are bitwise the unfused ones."""
    assert fused_logits[1] == logits[1]
    for schedule in SCHEDULES:
        for unfused, fused in zip(
            logits[0][schedule], fused_logits[0][schedule], strict=True
        ):
            np.testing.assert_array_equal(fused, unfused)
