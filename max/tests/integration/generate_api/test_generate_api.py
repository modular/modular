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
"""Checks the generate API tests' metrics and helpers on the CPU."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from _generate_api import (
    DATA_TARBALL_MTIME,
    Reference,
    check_near_tie,
    check_sha256,
    dump_reference,
    first_divergence,
    prefill_reference_steps,
    step_metrics,
    top_k_step,
    write_data_tarball,
)


def _logits(seed: int) -> npt.NDArray[np.float32]:
    return np.random.default_rng(seed).normal(0, 3, 1000).astype(np.float32)


def test_metrics_are_zero_for_identical_logits() -> None:
    logits = _logits(0)
    metrics = step_metrics(top_k_step(logits), logits)
    assert metrics.kl == pytest.approx(0.0, abs=1e-6)
    assert metrics.cos == pytest.approx(0.0, abs=1e-6)


def test_metrics_see_a_scaled_distribution() -> None:
    logits = _logits(1)
    metrics = step_metrics(top_k_step(logits), 2 * logits)
    assert metrics.kl > 0.1
    assert metrics.cos > 1e-3


def test_near_tie_allows_a_close_second_token() -> None:
    step = top_k_step(np.array([2.0, 1.8, -5.0], dtype=np.float32))
    check_near_tie(step, 1, "row 0", 0)


def test_near_tie_rejects_a_clear_winner() -> None:
    step = top_k_step(np.array([5.0, 1.0, -5.0], dtype=np.float32))
    with pytest.raises(AssertionError, match=r"row 2, step 3: .*near-tie"):
        check_near_tie(step, 1, "row 2", 3)


def test_near_tie_rejects_a_distant_token_after_a_tie() -> None:
    # The top two tie, but MAX picked the third token.
    step = top_k_step(np.array([2.0, 2.0, -5.0], dtype=np.float32))
    with pytest.raises(AssertionError, match="near-tie"):
        check_near_tie(step, 2, "batched row 0", 1)


def test_near_tie_rejects_a_token_outside_the_top_k() -> None:
    step = top_k_step(np.arange(64, dtype=np.float32))
    with pytest.raises(AssertionError, match="not in the reference's top"):
        check_near_tie(step, 0, "conversation 1", 4)


def test_first_divergence() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 4, 3], [1, 2, 3]) == 1
    with pytest.raises(ValueError):
        first_divergence([1, 2], [1, 2, 3])


@dataclasses.dataclass
class _Output:
    logits: list[npt.NDArray[np.float32]]


class _FakeModel:
    """Returns one logits row per call whose argmax is the context length."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def generate(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        top_k: int,
        ignore_eos: bool,
        output_logits: bool,
    ) -> _Output:
        assert (
            max_new_tokens == 1 and top_k == 1 and ignore_eos and output_logits
        )
        [ids] = input_ids
        self.calls.append(list(ids))
        row = np.zeros(32, dtype=np.float32)
        row[len(ids)] = 1.0
        return _Output([np.stack([row])])


def test_prefill_feeds_the_reference_tokens_after_a_divergence() -> None:
    model = _FakeModel()
    free_run = [np.full(32, -1.0, dtype=np.float32) for _ in range(4)]
    prefill_reference_steps(model, [7, 8], [10, 11, 12, 13], free_run, 1)
    assert model.calls == [[7, 8, 10, 11], [7, 8, 10, 11, 12]]
    assert int(free_run[2].argmax()) == 4
    assert int(free_run[3].argmax()) == 5
    # Steps up to the divergence keep their free-run logits.
    assert free_run[1].max() == -1.0


def test_dump_reference_writes_one_step_per_line() -> None:
    step = top_k_step(_logits(2))
    reference: Reference = {
        "model": "m",
        "revision": "r",
        "device": "d",
        "torch": "t",
        "transformers": "x",
        "attn_implementation": "sdpa",
        "top_k": 32,
        "rows": [
            {"prompt": "p", "prompt_ids": [1, 2], "steps": [step, step]},
            {
                "image": "dogs",
                "question": "q",
                "prompt_ids": [3],
                "steps": [step],
            },
        ],
    }
    text = dump_reference(reference)
    assert json.loads(text) == reference
    assert sum(json.dumps(step) in line for line in text.splitlines()) == 3


def test_data_tarball_is_reproducible(tmp_path: Path) -> None:
    files = {"references/b200/b.json": b"{}", "inputs/a.jpg": b"jpeg"}
    first = write_data_tarball(files, tmp_path / "first.tar.gz")
    second = write_data_tarball(
        dict(reversed(files.items())), tmp_path / "second.tar.gz"
    )
    assert first == second
    assert (
        first
        == hashlib.sha256((tmp_path / "first.tar.gz").read_bytes()).hexdigest()
    )
    with tarfile.open(tmp_path / "first.tar.gz") as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == sorted(files)
        for member in members:
            assert (member.mtime, member.mode, member.uid, member.gid) == (
                DATA_TARBALL_MTIME,
                0o644,
                0,
                0,
            )
            assert (member.uname, member.gname) == ("", "")
            extracted = tar.extractfile(member)
            assert extracted is not None
            assert extracted.read() == files[member.name]


def test_check_sha256_compares_with_the_s3_path(tmp_path: Path) -> None:
    path = tmp_path / "data.tar.gz"
    path.write_bytes(b"data")
    sha = hashlib.sha256(b"data").hexdigest()
    check_sha256(path, f"s3://bucket/artifacts/name/1/{sha}/data.tar.gz")
    with pytest.raises(AssertionError, match="sha256"):
        check_sha256(
            path, f"s3://bucket/artifacts/name/1/{'0' * 64}/data.tar.gz"
        )
