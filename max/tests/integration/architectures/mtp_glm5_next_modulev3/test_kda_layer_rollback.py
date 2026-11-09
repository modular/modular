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
"""One ModuleV3 KDA layer's verify and rollback against decoding the accepted
tokens.

The graph-API test's (``mtp_glm5_next/test_kda_layer_rollback.py``) on the
ModuleV3 layer. The speculative arm runs the layer over each request's
``K + 1`` window with the verify bound, then the rollback over the accepted
count. The base arm runs the same layer on its own copy of the pools, one
decode step per accepted token. Both pools must match exactly, which holds
only if the layer's decode pins its ``fma``s as the verify ring and the fold
do. Every projection is given exactly representable operands, so the row
count a matmul sees cannot change its result; the readouts pass through
``o_proj`` on non-dyadic inputs and are compared to a tolerance instead.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    ops,
)
from max.nn import Module
from max.nn._graph_ops.state_space import verify_width_operand
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.layers.kda import KdaReplayInputs
from max.pipelines.architectures.glm5_next.spec_state import (
    fold_kda_state_pools,
)
from max.pipelines.architectures.glm5_next.state_cache import (
    VERIFY_RING_LEAF_ID,
    state_regions,
    with_verify_ring,
)
from max.pipelines.architectures.glm5_next_modulev3.layers.kda import (
    KdaStateAccess,
    KdaVerify,
    KimiDeltaAttention,
)
from max.pipelines.architectures.glm5_next_modulev3.model_config import (
    Glm5NextConfig,
)
from max.pipelines.speculative.gated_delta_rollback import accepted_lengths
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh

HIDDEN = 256
HEADS = 2
HEAD_DIM = 128
KERNEL = 4
QKV_DIM = HEADS * HEAD_DIM
NUM_DRAFTS = 3
POOL_ROWS = 6
READOUT_REL_RMS = 1e-4

CONV_ROWS = (1, 4)
RECURRENT_ROWS = (5, 2)
RING_ROWS = (3, 0)
"""Each leaf's rows for the batch, distinct per leaf."""


def _ring_row_shape() -> tuple[int, ...]:
    regions = with_verify_ring(
        state_regions(
            num_kda_layers=1,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            conv_kernel_dim=KERNEL,
            num_sparse_layers=1,
            index_kpool=4,
            index_head_dim=128,
            dtype=DType.float32,
            num_devices=1,
        ),
        4,
    )
    (ring,) = (r for r in regions if r.leaf_id == VERIFY_RING_LEAF_ID)
    return ring.row_shape


def _sparse(
    rng: np.random.Generator, out_dim: int, in_dim: int, scale: float
) -> np.ndarray:
    """Returns a weight with four entries of ``+-scale`` per output row."""
    weight = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row in range(out_dim):
        cols = rng.choice(in_dim, size=4, replace=False)
        weight[row, cols] = rng.choice([-scale, scale], size=4)
    return weight


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(29)

    def conv() -> np.ndarray:
        # The checkpoint's depthwise `[channels, 1, kernel]`.
        return (
            (rng.integers(-4, 5, size=(QKV_DIM, KERNEL)) / 8)
            .astype(np.float32)
            .reshape(QKV_DIM, 1, KERNEL)
        )

    weights = {
        "q_proj.weight": _sparse(rng, QKV_DIM, HIDDEN, 0.5),
        "k_proj.weight": _sparse(rng, QKV_DIM, HIDDEN, 0.5),
        "v_proj.weight": _sparse(rng, QKV_DIM, HIDDEN, 0.5),
        "q_conv1d": conv(),
        "k_conv1d": conv(),
        "v_conv1d": conv(),
        "f_a_proj.weight": _sparse(rng, HEAD_DIM, HIDDEN, 0.25),
        "f_b_proj.weight": _sparse(rng, QKV_DIM, HEAD_DIM, 0.25),
        "dt_bias": (rng.integers(-8, 9, size=QKV_DIM) / 16).astype(np.float32),
        "A_log": (rng.integers(-4, 5, size=HEADS) / 8).astype(np.float32),
        "b_proj.weight": _sparse(rng, HEADS, HIDDEN, 0.5),
        "g_a_proj.weight": _sparse(rng, HEAD_DIM, HIDDEN, 0.25),
        "g_b_proj.weight": _sparse(rng, QKV_DIM, HEAD_DIM, 0.25),
        "o_norm.weight": np.ones(HEAD_DIM, dtype=np.float32),
        "o_proj.weight": _sparse(rng, HIDDEN, QKV_DIM, 0.5),
    }
    return {f"layer.{name}": value for name, value in weights.items()}


def _config() -> Glm5NextConfig:
    devices = [DeviceRef.GPU()]
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        page_size=128,
        devices=devices,
        num_q_heads=8,
    )
    return Glm5NextConfig(
        dtype=DType.float32,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=1024,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-5,
        linear_num_heads=HEADS,
        linear_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=KERNEL,
        linear_lower_bound=-5.0,
    )


def _pools() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(13)
    conv = rng.integers(-2, 3, size=(POOL_ROWS, 3 * QKV_DIM, KERNEL - 1))
    recurrent = rng.standard_normal((POOL_ROWS, HEADS, HEAD_DIM, HEAD_DIM))
    return {
        "conv": conv.astype(np.float32),
        "recurrent": (0.1 * recurrent).astype(np.float32),
    }


def _table(rows: tuple[int, ...]) -> np.ndarray:
    return np.array([rows], dtype=np.uint32)


def _offsets(lengths: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(
        np.sqrt(((a64 - b64) ** 2).mean()) / (np.sqrt((a64**2).mean()) + 1e-12)
    )


def _graph_value(value: Tensor) -> Any:
    (graph_value,) = value.graph_values
    return graph_value


class _Rollback(Module[..., tuple[Tensor, ...]]):
    """The verify and rollback beside the decode steps, in one graph."""

    def __init__(self, layer: KimiDeltaAttention, num_steps: int) -> None:
        layer.pin_decode_fma = True
        self.layer = layer
        self.num_steps = num_steps

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        x, offsets, accepted, conv_rows, rec_rows, ring_rows = inputs[:6]
        steps = inputs[6:-5]
        conv, rec, ring, base_conv, base_rec = inputs[-5:]
        mesh = self.layer.o_proj.weight.mesh
        mapping = DeviceMapping.replicated(mesh)

        def dist(value: Tensor) -> Tensor:
            return Tensor.from_shard_values([_graph_value(value)], mapping)

        layer_idx = F.constant(0, DType.int64, device=CPU())

        def state(conv_pool: Tensor, rec_pool: Tensor) -> KdaStateAccess:
            return KdaStateAccess(
                conv_pool=dist(conv_pool),
                conv_rows=dist(conv_rows),
                recurrent_pool=dist(rec_pool),
                recurrent_rows=dist(rec_rows),
                layer=layer_idx,
            )

        verify_width = verify_width_operand(NUM_DRAFTS)
        captures: list[KdaReplayInputs] = []
        self.layer.verify = KdaVerify(
            ring_pool=dist(ring),
            ring_rows=dist(ring_rows),
            verify_width=verify_width,
            captures=[captures],
        )
        spec_out = self.layer(dist(x), state(conv, rec), dist(offsets))
        self.layer.verify = None

        gpu = DeviceRef.GPU()
        offsets_v = _graph_value(offsets)
        accepted_v = _graph_value(accepted)
        conv_v, rec_v, ring_v = (_graph_value(b) for b in (conv, rec, ring))
        assert isinstance(conv_v, BufferValue)
        num_drafts = ops.constant(NUM_DRAFTS, DType.int64, device=gpu)
        fold_kda_state_pools(
            [rec_v],
            [_graph_value(rec_rows)],
            [ring_v],
            [_graph_value(ring_rows)],
            [conv_v],
            [_graph_value(conv_rows)],
            [captures],
            accepted_lengths(offsets_v, accepted_v, num_drafts),
            [offsets_v],
            [],
            verify_width,
        )
        base_outs = [
            self.layer(
                dist(steps[2 * t]),
                state(base_conv, base_rec),
                dist(steps[2 * t + 1]),
            )
            for t in range(self.num_steps)
        ]
        return (spec_out, *base_outs)


@functools.cache
def _run_spec_against_decode(
    accepted: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Runs the verify and rollback, then the accepted tokens one at a time.

    Returns the pools, and each accepted token's readout from both arms.
    Cached, so a test reusing another's accepted counts compiles nothing.
    """
    batch = len(accepted)
    window = NUM_DRAFTS + 1
    offsets = _offsets([window] * batch)
    rng = np.random.default_rng(31)
    x = rng.integers(-1, 2, size=(window * batch, HIDDEN)).astype(np.float32)

    # Step ``t`` decodes token ``t`` of every request that accepted it.
    steps = []
    for t in range(1 + max(accepted)):
        members = [b for b in range(batch) if t <= accepted[b]]
        steps.append(
            (
                x[[b * window + t for b in members]],
                _offsets([int(t <= accepted[b]) for b in range(batch)]),
                members,
            )
        )

    device = Accelerator()
    gpu = DeviceRef.GPU()
    table = TensorType(DType.uint32, [1, "batch_size"], device=gpu)
    pools = _pools()
    ring_shape = (POOL_ROWS, *_ring_row_shape())
    types: list[TensorType | BufferType] = [
        TensorType(DType.float32, list(x.shape), device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        table,
        table,
        table,
    ]
    for step_x, _, _ in steps:
        types.append(TensorType(DType.float32, list(step_x.shape), device=gpu))
        types.append(TensorType(DType.uint32, [batch + 1], device=gpu))
    conv_type = BufferType(DType.float32, pools["conv"].shape, device=gpu)
    rec_type = BufferType(DType.float32, pools["recurrent"].shape, device=gpu)
    types += [
        conv_type,
        rec_type,
        BufferType(DType.float32, ring_shape, device=gpu),
        conv_type,
        rec_type,
    ]

    mesh = DeviceMesh((device,), (1,), ("tp",))
    with F.lazy(), default_dtype(DType.float32), default_device(mesh):
        module = _Rollback(KimiDeltaAttention(_config()), len(steps))
    compiled = module.compile(*types, weights=_weights())

    def put(value: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(value)).to(device)

    buffers = {
        "conv": put(pools["conv"]),
        "recurrent": put(pools["recurrent"]),
        "ring": put(np.zeros(ring_shape, dtype=np.float32)),
        "base_conv": put(pools["conv"]),
        "base_recurrent": put(pools["recurrent"]),
    }
    step_buffers = []
    for step_x, step_offsets, _ in steps:
        step_buffers += [put(step_x), put(step_offsets)]
    outputs = compiled(
        put(x),
        put(offsets),
        put(np.array(accepted, dtype=np.int64)),
        put(_table(CONV_ROWS)),
        put(_table(RECURRENT_ROWS)),
        put(_table(RING_ROWS)),
        *step_buffers,
        *buffers.values(),
    )
    host = [
        np.asarray(out.to(CPU()).driver_tensor.to_numpy()) for out in outputs
    ]
    spec, base = host[0], host[1:]
    readouts = [
        (spec[b * window + t], base[t][i])
        for t, (_, _, members) in enumerate(steps)
        for i, b in enumerate(members)
    ]
    return (
        {
            name: np.array(buf.to(CPU()).to_numpy())
            for name, buf in buffers.items()
        },
        readouts,
    )


@pytest.mark.parametrize(
    "accepted", [(0, 0), (1, 1), (NUM_DRAFTS, NUM_DRAFTS), (NUM_DRAFTS, 0)]
)
def test_the_rollback_lands_where_decoding_the_accepted_tokens_does(
    accepted: tuple[int, ...],
) -> None:
    pools, readouts = _run_spec_against_decode(accepted)
    np.testing.assert_array_equal(pools["conv"], pools["base_conv"])
    np.testing.assert_array_equal(pools["recurrent"], pools["base_recurrent"])
    spec = np.stack([s for s, _ in readouts])
    base = np.stack([b for _, b in readouts])
    assert _rel_rms(base, spec) < READOUT_REL_RMS


def test_a_rejected_draft_changes_the_state() -> None:
    """Checks the fixture can tell an accepted count from another."""
    rejected, _ = _run_spec_against_decode((0, 0))
    accepted, _ = _run_spec_against_decode((NUM_DRAFTS, NUM_DRAFTS))
    assert not np.array_equal(rejected["recurrent"], accepted["recurrent"])
    assert not np.array_equal(rejected["conv"], accepted["conv"])
