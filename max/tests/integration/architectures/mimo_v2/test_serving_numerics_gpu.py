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
and the MAX model, tensor parallel over two GPUs when there are two, and
compares every logit row with the goldens that ``make_random_fixture.py
--tiny-goldens`` computed from the checkpoint's own HF modeling code in
float32. Both prompts
run as one ragged batch, prefilled in chunks through the paged KV cache and
then decoded a token at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiny_fixture
from max.driver import accelerator_count
from mimo_runner import Runner

GOLDENS = Path(__file__).parent / "testdata" / "tiny_goldens.npz"
CHUNK, DECODE = 96, 4

MEAN_KL, MAX_KL, TOP1 = 4e-3, 2.5e-2, 0.92
"""Bounds on KL(HF || MAX) over the whole vocabulary, and on top-1
agreement, over every row of both prompts.

A correct run on two B200s gives a mean KL of 1.4e-3, a maximum of 6.6e-3
and top-1 agreement of 0.952. The goldens decode the same quantized weights
in float32, while MAX also quantizes the GEMM inputs: FP8 for the dense
projections and MXFP4 for the experts. The random logits are flat (a mean
top probability of 0.12), so near-ties cost top-1. The bounds sit at 2.8x
and 3.8x the clean KL and three points of top-1 below it. Each of these
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


def test_serving_logits_match_hf(tmp_path: Path) -> None:
    goldens = np.load(GOLDENS)
    checkpoint = tiny_fixture.tensors()
    assert str(goldens["digest"]) == tiny_fixture.digest(checkpoint), (
        "tiny_fixture.py no longer builds the checkpoint the goldens were "
        "made from; rerun make_random_fixture.py --tiny-goldens."
    )
    tiny_fixture.write(tmp_path, checkpoint)
    num_devices = min(2, accelerator_count())
    runner = Runner(
        tmp_path,
        num_devices=num_devices,
        max_seq_len=256,
        kv_cache_bytes=256 * 1024**2,
    )
    prompts = tiny_fixture.prompts()
    outputs = runner.teacher_forced(
        [np.array(ids, np.int64) for ids in prompts.values()], CHUNK, DECODE
    )

    kl, top1 = {}, {}
    for name, (logits, _) in zip(prompts, outputs, strict=True):
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
