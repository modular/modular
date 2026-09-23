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
"""The ring-state short conv ops against a full-history numpy oracle.

One graph runs three conv sites with their commits over a ragged batch with
symbolic dims. The oracle keeps every accepted input per sequence and
convolves against all of it. Scenarios: ragged prefill, decode, verify-style
chunks with rollback, chunks shorter than the tap count, chunks longer than
the ring, and decode beside prefill.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type
from max.nn.state_space import short_conv_ring_commit, short_conv_ring_fwd

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU"
)

WIDTH = 4
NUM_DRAFT_TOKENS = 3
RING_LEN = WIDTH - 1 + NUM_DRAFT_TOKENS
KV_DIM = 64
# Not a multiple of the 256-channel block, so the channel guard runs.
RES_DIM = 300
MAX_ROWS = 5
DTYPE = DType.bfloat16
RTOL = 2e-2
ATOL = 1e-2
GPU = DeviceRef.GPU()

SITES = ("k", "v", "res")
CHANNELS = {"k": KV_DIM, "v": KV_DIM, "res": RES_DIM}


def _to_bf16(arr: np.ndarray) -> Buffer:
    bits = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16)


def _from_bf16(buf: Buffer) -> np.ndarray:
    return (
        buf.copy(device=CPU())
        .view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def _round_bf16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(ml_dtypes.bfloat16).astype(np.float32)


def _build_graph() -> Graph:
    # The state cache names each ring's slot count after its own leaf, so
    # no two rings share a symbolic slot dim.
    input_types: list[Type[Any]] = [
        TensorType(DType.uint32, ["batch_plus_1"], GPU),
        TensorType(DType.uint32, ["total_seq_len"], GPU),
        TensorType(DType.uint32, ["batch"], GPU),
    ]
    for site in SITES:
        channels = CHANNELS[site]
        input_types += [
            TensorType(DTYPE, ["total_seq_len", channels], GPU),
            TensorType(DTYPE, [channels, WIDTH], GPU),
            BufferType(
                DType.float32, [f"{site}_rows", RING_LEN, channels], GPU
            ),
        ]

    with Graph("short_conv_ring", input_types=input_types) as g:
        offsets, positions, rows = (v.tensor for v in g.inputs[:3])
        outs = []
        for i in range(len(SITES)):
            x, weight = (v.tensor for v in g.inputs[3 + 3 * i : 5 + 3 * i])
            ring = g.inputs[5 + 3 * i].buffer
            outs.append(
                short_conv_ring_fwd(x, weight, ring, offsets, positions, rows)
            )
            short_conv_ring_commit(x, ring, offsets, positions, rows)
        g.output(*outs)
    return g


def _conv_with_history(
    chunk: np.ndarray, history: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """`x + conv(x)` for one chunk given every input before it; missing
    history reads as zero."""
    pad = np.zeros(
        (max(0, WIDTH - 1 - len(history)), chunk.shape[1]), np.float32
    )
    full = np.concatenate([pad, history[-(WIDTH - 1) :], chunk], axis=0)
    out = chunk.copy()
    for j in range(WIDTH):
        # weight[:, WIDTH - 1 - j] multiplies the input j positions back.
        taps = full[WIDTH - 1 - j : WIDTH - 1 - j + len(chunk)]
        out += taps * weight[:, WIDTH - 1 - j]
    return out


@dataclasses.dataclass
class _Sequence:
    """One request's accepted inputs so far, per conv site."""

    slot: int
    history: dict[str, np.ndarray] = dataclasses.field(
        default_factory=lambda: {
            site: np.zeros((0, CHANNELS[site]), np.float32) for site in SITES
        }
    )

    @property
    def position(self) -> int:
        return len(self.history["k"])

    def rollback(self, rejected: int) -> None:
        """Drops the last `rejected` inputs, as verify does."""
        if rejected:
            for site in SITES:
                self.history[site] = self.history[site][:-rejected]


class _Harness:
    """Drives the compiled graph against the oracle, one step at a time."""

    def __init__(self, session: InferenceSession, seed: int = 7) -> None:
        self.model = session.load(_build_graph())
        self.device = self.model.input_devices[0]
        self.rng = np.random.default_rng(seed)

        self.weights = {
            site: _round_bf16(
                self.rng.standard_normal((CHANNELS[site], WIDTH)).astype(
                    np.float32
                )
            )
            for site in SITES
        }
        self.weight_bufs = {
            site: _to_bf16(w).to(self.device)
            for site, w in self.weights.items()
        }
        # Garbage, so stale entries are never mistaken for history.
        self.rings = {
            site: Buffer.from_numpy(
                self.rng.standard_normal(
                    (MAX_ROWS, RING_LEN, CHANNELS[site])
                ).astype(np.float32)
                * 100
            ).to(self.device)
            for site in SITES
        }

    def step(self, seqs: Sequence[_Sequence], lengths: Sequence[int]) -> None:
        """One forward of `lengths[i]` tokens per sequence; checks every
        site's output and ring."""
        total = sum(lengths)
        offsets = np.cumsum([0, *lengths]).astype(np.uint32)
        positions = np.concatenate(
            [
                np.arange(seq.position, seq.position + n)
                for seq, n in zip(seqs, lengths, strict=True)
            ]
        ).astype(np.uint32)
        rows = np.array([seq.slot for seq in seqs], np.uint32)
        xs = {
            site: _round_bf16(
                self.rng.standard_normal((total, CHANNELS[site])).astype(
                    np.float32
                )
            )
            for site in SITES
        }

        args: list[Any] = [
            Buffer.from_numpy(offsets).to(self.device),
            Buffer.from_numpy(positions).to(self.device),
            Buffer.from_numpy(rows).to(self.device),
        ]
        for site in SITES:
            args += [
                _to_bf16(xs[site]).to(self.device),
                self.weight_bufs[site],
                self.rings[site],
            ]

        got = self.model.execute(*args)
        outs = {}
        for site, out in zip(SITES, got, strict=True):
            assert isinstance(out, Buffer)
            outs[site] = _from_bf16(out)
        rings = {
            site: ring.copy(device=CPU()).to_numpy()
            for site, ring in self.rings.items()
        }

        for b, (seq, n) in enumerate(zip(seqs, lengths, strict=True)):
            sl = slice(int(offsets[b]), int(offsets[b + 1]))
            for site in SITES:
                ref = _round_bf16(
                    _conv_with_history(
                        xs[site][sl], seq.history[site], self.weights[site]
                    )
                )
                np.testing.assert_allclose(
                    outs[site][sl],
                    ref,
                    rtol=RTOL,
                    atol=ATOL,
                    err_msg=f"{site} seq {b}",
                )

            for site in SITES:
                seq.history[site] = np.concatenate(
                    [seq.history[site], xs[site][sl]]
                )

            # Every position the next chunk can read owns its entry: at most
            # min(n - 1, k) of this chunk can be rejected, and the first tap
            # reaches WIDTH - 1 further back.
            end = seq.position
            reach = min(n - 1, NUM_DRAFT_TOKENS) + WIDTH - 1
            for pos in range(max(0, end - reach), end):
                for site in SITES:
                    np.testing.assert_array_equal(
                        rings[site][seq.slot, pos % RING_LEN],
                        seq.history[site][pos],
                        err_msg=f"{site} ring seq {b} pos {pos}",
                    )


@pytest.fixture(scope="module")
def harness() -> _Harness:
    return _Harness(InferenceSession(devices=[Accelerator()]))


def test_prefill_then_decode(harness: _Harness) -> None:
    """Ragged prefill from position zero, then single tokens. The slots
    start out holding garbage, so this also checks that taps before position
    zero read zero rather than the slot."""
    seqs = [_Sequence(slot=2), _Sequence(slot=0), _Sequence(slot=4)]
    harness.step(seqs, [5, 1, 9])
    harness.step(seqs, [1, 1, 1])
    harness.step(seqs, [1, 1, 1])


def test_verify_chunk_then_rollback(harness: _Harness) -> None:
    """Verify commits every draft position; after rejections the next chunk
    must read accepted history, not the rejected inputs."""
    seqs = [_Sequence(slot=0), _Sequence(slot=1), _Sequence(slot=2)]
    harness.step(seqs, [7, 3, 12])
    # Bonus token plus NUM_DRAFT_TOKENS drafts per sequence.
    harness.step(seqs, [NUM_DRAFT_TOKENS + 1] * 3)
    # Reject 0, 2 and every draft token respectively.
    for seq, rejected in zip(seqs, (0, 2, NUM_DRAFT_TOKENS), strict=True):
        seq.rollback(rejected)
    harness.step(seqs, [NUM_DRAFT_TOKENS + 1] * 3)
    for seq, rejected in zip(seqs, (1, 3, 0), strict=True):
        seq.rollback(rejected)
    harness.step(seqs, [1, 1, 1])


def test_chunk_shorter_than_the_taps(harness: _Harness) -> None:
    """Chunks shorter than the tap count mix ring history with chunk input."""
    seqs = [_Sequence(slot=3), _Sequence(slot=4)]
    harness.step(seqs, [4, 6])
    harness.step(seqs, [2, 2])
    harness.step(seqs, [2, 1])
    harness.step(seqs, [1, 2])


def test_chunk_longer_than_the_ring(harness: _Harness) -> None:
    """Only a long chunk's tail is committed; the next step must still see
    the right taps."""
    seqs = [_Sequence(slot=0), _Sequence(slot=2)]
    harness.step(seqs, [RING_LEN + 5, 2])
    harness.step(seqs, [1, RING_LEN * 3 + 1])
    harness.step(seqs, [1, 1])
    harness.step(seqs, [RING_LEN, RING_LEN + 1])
    harness.step(seqs, [1, 1])


def test_mixed_batch_decode_beside_prefill(harness: _Harness) -> None:
    """One extending sequence beside decoding ones."""
    seqs = [_Sequence(slot=0), _Sequence(slot=1), _Sequence(slot=2)]
    harness.step(seqs, [6, 6, 6])
    harness.step(seqs, [1, 9, 1])
    harness.step(seqs, [1, 1, 30])
    harness.step(seqs, [1, 1, 1])
