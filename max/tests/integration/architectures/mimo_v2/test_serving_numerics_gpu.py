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
"""Scores MAX's MiMo-V2 logits on the tiny fixture against HF goldens.

Builds the ``tiny_fixture`` checkpoint, runs it through the weight adapter
and the MAX model, tensor parallel over the target's selected GPU count, and
compares every logit row with the goldens that ``make_random_fixture.py
--tiny-goldens`` computed from the checkpoint's own HF modeling code in
float32. Both prompts
run as one ragged batch, prefilled in chunks through the paged KV cache and
then decoded a token at a time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tiny_fixture
from max.driver import accelerator_count
from mimo_runner import Runner

GOLDENS = Path(__file__).parent / "testdata" / "tiny_goldens.npz"
CHUNK, DECODE = 96, 4

MEAN_KL, MAX_KL, TOP1 = 4e-3, 2.5e-2, 0.92
"""Bounds on KL(HF || MAX) over the whole vocabulary, and on top-1
agreement, over every row of both prompts.

The original MXFP4-activation run on two B200s gave mean KL 1.4e-3,
maximum KL 6.6e-3 and top-1 agreement 0.952. With MXFP8 expert inputs, mean
KL is 1.09e-3, maximum KL 6.6e-3 and top-1 agreement 0.972. These measurements
are TP2 eager with unfused QKV preparation. The goldens
decode the same quantized weights in float32; GEMM inputs are FP8 for the
dense projections and MXFP8 for the experts. The random logits are flat
(a mean top probability of 0.12), so near-ties cost top-1. The bounds pass
both runs; test_expert_precision_gpu.py pins the expert precision. Each of these
defects exceeds a bound by at least 6x: no sinks (mean 2.6e-2), a 129-key
window (row 148, 3.5e-1), RoPE over the whole head (mean 1.85e-1) and
interleaved RoPE (mean 2.9e-2)."""


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
def logits_by_mode(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[bool, bool, bool], list[npt.NDArray[np.float32]]]:
    checkpoint = tiny_fixture.tensors()
    goldens = np.load(GOLDENS)
    assert str(goldens["digest"]) == tiny_fixture.digest(checkpoint), (
        "tiny_fixture.py no longer builds the checkpoint the goldens were "
        "made from; rerun make_random_fixture.py --tiny-goldens."
    )
    path = tmp_path_factory.mktemp("mimo")
    tiny_fixture.write(path, checkpoint)
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    assert accelerator_count() >= num_devices
    prompts = [
        np.array(ids, np.int64) for ids in tiny_fixture.prompts().values()
    ]
    results = {}
    for sliding, full in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        model = Runner(
            path,
            num_devices=num_devices,
            max_seq_len=256,
            kv_cache_bytes=256 * 1024**2,
            fused_sliding_qkv=sliding,
            fused_full_qkv=full,
        )
        assert (model.fused_qkv_count > 0) == (sliding or full)
        for captured in (False, True):
            outputs = model.teacher_forced(
                prompts, CHUNK, DECODE, device_graph_capture=captured
            )
            assert (model.capture_steps > 0) == captured
            if captured:
                assert model.capture_steps == len(
                    model.capture_replays_by_batch_size
                )
                assert model.replay_steps > model.capture_steps
                assert any(
                    replays > 1
                    for replays in model.capture_replays_by_batch_size.values()
                )
                assert any(
                    len(set(lengths)) > 1
                    for lengths in model.replay_cache_lengths_by_batch_size.values()
                )
            else:
                assert model.replay_steps == 0
            logits = []
            for output, _ in outputs:
                assert output is not None
                logits.append(output)
            results[sliding, full, captured] = logits
        del model
    return results


@pytest.mark.parametrize(
    ("sliding", "full"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["unfused", "sliding", "full", "combined"],
)
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "capture"])
def test_serving_logits_match_hf(
    logits_by_mode: dict[
        tuple[bool, bool, bool], list[npt.NDArray[np.float32]]
    ],
    sliding: bool,
    full: bool,
    captured: bool,
) -> None:
    goldens = np.load(GOLDENS)
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    prompts = tiny_fixture.prompts()
    outputs = logits_by_mode[sliding, full, captured]
    kl, top1 = {}, {}
    for name, logits in zip(prompts, outputs, strict=True):
        assert logits is not None
        gold = goldens[f"logits_{name}"]
        kl[name] = _kl(gold, logits)
        top1[name] = gold.argmax(-1) == logits.argmax(-1)
    rows = np.concatenate(list(kl.values()))
    agree = np.concatenate(list(top1.values())).mean()
    edge = tiny_fixture.BEACON_ROW + tiny_fixture.SLIDING_WINDOW
    summary = (
        f"{num_devices} GPU(s): KL mean {rows.mean():.2e}, max "
        f"{rows.max():.2e} (row {edge} of the long prompt "
        f"{kl['long'][edge]:.2e}); top-1 {agree:.3f}"
    )
    print(summary)
    assert rows.mean() <= MEAN_KL, summary
    assert rows.max() <= MAX_KL, summary
    assert agree >= TOP1, summary


@pytest.mark.parametrize(
    ("sliding", "full"),
    [(True, False), (False, True), (True, True)],
    ids=["sliding", "full", "combined"],
)
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "capture"])
def test_fused_logits_match_unfused(
    sliding: bool,
    full: bool,
    logits_by_mode: dict[
        tuple[bool, bool, bool], list[npt.NDArray[np.float32]]
    ],
    captured: bool,
) -> None:
    for unfused, fused in zip(
        logits_by_mode[False, False, captured],
        logits_by_mode[sliding, full, captured],
        strict=True,
    ):
        np.testing.assert_array_equal(fused, unfused)
