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
"""Pooled indexer keys across a long speculative schedule, past 2048 tokens.

Each request prefills past 2048 tokens, then verifies windows of ``K + 1``
keys in which every draft after the accepted ones carries a wrong key. The
pooled-key writers run as :meth:`Indexer._select_pooled` runs them, with the
verify's tail seed deferred and replayed over the accepted rows. Every pool
the accepted sequence completes must come out bit-identical to compressing
that sequence in one shot, however many times a rejected draft wrote it on
the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.legacy.kernels import (
    mla_kpool_compress,
    mla_kpool_ring_close,
    mla_kpool_seed_tail,
)
from max.pipelines.architectures.deepseekV3_2.layers.indexer import (
    KpoolTailReplayInputs,
)
from max.pipelines.architectures.glm5_next.spec_state import (
    deferred_seed_offsets,
    replay_tail_seeds,
)
from max.pipelines.speculative.gated_delta_rollback import (
    accepted_row_plan,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar

KPOOL = 4
HEAD_DIM = 128
NUM_DRAFTS = 3
PREFILL = (2049, 2054, 2071)
"""Past 2048 and at three different pool remainders."""
DECODE_STEPS = 48
SLOTS = (4, 0, 2)
RING_SHAPE = (6, 2, KPOOL, HEAD_DIM)


def _bf16(values: np.ndarray) -> np.ndarray:
    """Returns float32 values truncated to bfloat16, as uint16 bits."""
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _host(value: np.ndarray, device: Accelerator) -> Buffer:
    buffer = Buffer.from_numpy(np.ascontiguousarray(value))
    if value.dtype == np.uint16:
        buffer = buffer.view(DType.bfloat16)
    return buffer.to(device)


def _bits(buffer: object) -> np.ndarray:
    assert isinstance(buffer, Buffer)
    host = buffer.to(CPU())
    if host.dtype == DType.bfloat16:
        host = host.view(DType.uint16)
    return np.array(host.to_numpy())


@dataclass
class _Rig:
    device: Accelerator
    step: Model
    oracle: Model


def _compile(defer: bool) -> _Rig:
    gpu = DeviceRef.GPU()
    rows = TensorType(DType.bfloat16, ["rows", HEAD_DIM], gpu)
    ape = TensorType(DType.float32, [KPOOL, HEAD_DIM], gpu)
    offsets = TensorType(DType.uint32, ["batch_plus_one"], gpu)
    batch = TensorType(DType.uint32, ["batch"], gpu)
    with Graph(
        "speculative_kpool_step",
        input_types=[
            rows,
            rows,
            ape,
            offsets,
            offsets,
            batch,
            batch,
            TensorType(DType.int64, ["batch"], gpu),
            # Carries the verify width as its second dim; never read.
            TensorType(DType.int64, ["batch", "num_steps"], gpu),
            BufferType(DType.bfloat16, RING_SHAPE, gpu),
        ],
    ) as graph:
        k, gate, ape_v, offs, pool_offs, clen, slots, accepted, drafts = (
            value.tensor for value in graph.inputs[:9]
        )
        tail = graph.inputs[9].buffer
        num_drafts = drafts.shape[1]
        ring_pooled, ring_closed = mla_kpool_ring_close(
            tail, k, gate, ape_v, offs, clen, slots, KPOOL
        )
        compressed = mla_kpool_compress(
            k, gate, ape_v, offs, pool_offs, clen, KPOOL
        )
        (seed,) = deferred_seed_offsets(num_drafts, [offs]) if defer else [offs]
        mla_kpool_seed_tail(tail, k, gate, seed, clen, slots, KPOOL)
        k_scalar = _shape_to_scalar(num_drafts, gpu)
        row_indices, replay_offsets = accepted_row_plan(
            offs, accepted, k_scalar, k.shape[0], gpu
        )
        replay_tail_seeds(
            [
                [
                    KpoolTailReplayInputs(
                        k=k,
                        gate=gate,
                        tail=tail,
                        slot_idx=slots,
                        cache_lengths=clen,
                        kpool=KPOOL,
                    )
                ]
            ],
            row_indices,
            replay_offsets,
            k_scalar,
            [],
        )
        graph.output(ring_pooled, ring_closed, compressed)

    with Graph(
        "kpool_oracle", input_types=[rows, rows, ape, offsets, offsets, batch]
    ) as oracle:
        k, gate, ape_v, offs, pool_offs, clen = (
            value.tensor for value in oracle.inputs
        )
        oracle.output(
            mla_kpool_compress(k, gate, ape_v, offs, pool_offs, clen, KPOOL)
        )

    device = Accelerator()
    session = InferenceSession(devices=[device])
    return _Rig(device, session.load(graph), session.load(oracle))


def _offsets(counts: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)


def _run(defer: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Runs the schedule; returns each request's written and oracle pools."""
    rig = _compile(defer)
    rng = np.random.default_rng(17)
    batch = len(PREFILL)
    max_len = max(PREFILL) + DECODE_STEPS * (NUM_DRAFTS + 1) + 8
    true_k = [_bf16(rng.standard_normal((max_len, HEAD_DIM))) for _ in PREFILL]
    true_g = [
        _bf16(4 * rng.standard_normal((max_len, HEAD_DIM))) for _ in PREFILL
    ]
    ape = (2 * rng.standard_normal((KPOOL, HEAD_DIM))).astype(np.float32)
    ring = _host(np.zeros(RING_SHAPE, dtype=np.uint16), rig.device)
    written = [
        np.zeros((max_len // KPOOL, HEAD_DIM), np.uint16) for _ in PREFILL
    ]
    positions = [0] * batch
    slots = np.array(SLOTS, dtype=np.uint32)

    def step(
        keys: list[np.ndarray],
        gates: list[np.ndarray],
        k: int,
        acc: list[int],
    ) -> None:
        counts = [len(x) for x in keys]
        clen = np.array(positions, dtype=np.uint32).astype(np.int64)
        n = np.array(counts, dtype=np.int64)
        full = np.maximum((clen + n) // KPOOL - (clen + KPOOL - 1) // KPOOL, 0)
        pool_offsets = _offsets(list(full))
        outputs = rig.step.execute(
            _host(np.concatenate(keys), rig.device),
            _host(np.concatenate(gates), rig.device),
            _host(ape, rig.device),
            _host(_offsets(counts), rig.device),
            _host(pool_offsets, rig.device),
            _host(clen.astype(np.uint32), rig.device),
            _host(slots, rig.device),
            _host(np.array(acc, dtype=np.int64), rig.device),
            _host(np.zeros((batch, k), dtype=np.int64), rig.device),
            ring,
        )
        rp, rc, cp = (_bits(out) for out in outputs)
        for b in range(batch):
            if int(rc[b]) >= 0:
                written[b][int(rc[b])] = rp[b]
            first = (int(clen[b]) + KPOOL - 1) // KPOOL
            for p in range(int(full[b])):
                written[b][first + p] = cp[int(pool_offsets[b]) + p]

    # Prefill: no drafts, every row committed.
    step(
        [true_k[b][:p] for b, p in enumerate(PREFILL)],
        [true_g[b][:p] for b, p in enumerate(PREFILL)],
        0,
        [0] * batch,
    )
    for b, p in enumerate(PREFILL):
        positions[b] = p

    for _ in range(DECODE_STEPS):
        accepted = [int(a) for a in rng.integers(0, NUM_DRAFTS + 1, batch)]
        keys, gates = [], []
        for b in range(batch):
            start = positions[b]
            keep = 1 + accepted[b]
            # The window's committed rows carry the true keys; every draft
            # past the accepted ones carries a key the sequence never has.
            k_win = true_k[b][start : start + NUM_DRAFTS + 1].copy()
            g_win = true_g[b][start : start + NUM_DRAFTS + 1].copy()
            k_win[keep:] = _bf16(
                rng.standard_normal((len(k_win) - keep, HEAD_DIM))
            )
            g_win[keep:] = _bf16(
                4 * rng.standard_normal((len(g_win) - keep, HEAD_DIM))
            )
            keys.append(k_win)
            gates.append(g_win)
        step(keys, gates, NUM_DRAFTS, accepted)
        for b in range(batch):
            positions[b] += 1 + accepted[b]

    oracle = []
    for b in range(batch):
        length = positions[b]
        pools = length // KPOOL
        (out,) = rig.oracle.execute(
            _host(true_k[b][:length], rig.device),
            _host(true_g[b][:length], rig.device),
            _host(ape, rig.device),
            _host(_offsets([length]), rig.device),
            _host(_offsets([pools]), rig.device),
            _host(np.zeros(1, dtype=np.uint32), rig.device),
        )
        oracle.append(_bits(out)[:pools])
    return [w[: len(o)] for w, o in zip(written, oracle, strict=True)], oracle


def test_every_accepted_pool_matches_one_shot_compression() -> None:
    """Checks every pool past 2048, rejections included, is bit-identical."""
    written, oracle = _run(defer=True)
    for b, (w, o) in enumerate(zip(written, oracle, strict=True)):
        assert len(o) * KPOOL > 2048 + 32, "the schedule must run past 2048"
        bad = np.flatnonzero((w != o).any(axis=1))
        assert bad.size == 0, f"request {b}: pools {bad[:8].tolist()} differ"


def test_seeding_rejected_drafts_is_detectable() -> None:
    """Checks a verify that seeds its drafts corrupts some pool."""
    written, oracle = _run(defer=False)
    assert any((w != o).any() for w, o in zip(written, oracle, strict=True)), (
        "the schedule never exercised a pending member a draft could clobber"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-q"]))
