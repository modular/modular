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
"""The ``mamba2_replay_fold`` op lands the states a per-layer replay would.

The reference replays each layer's accepted rows through the conv and scan
ops themselves, gathered into one ragged batch with cumulative offsets. The
fold addresses each request's ``(start, end)`` rows in place. Both arms run in
one graph on copies of the same pools, which must come out bitwise equal, for
a decode-sized Nemotron-H layout with ``dstate`` 128.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.state_space import (
    causal_conv1d_varlen_fwd,
    mamba2_replay_fold,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)

_LAYERS = 3
_NHEADS = 8
_HEAD_DIM = 64
_NGROUPS = 2
_DSTATE = 128
_WIDTH = 4
_INTERMEDIATE = _NHEADS * _HEAD_DIM
_CONV_DIM = _INTERMEDIATE + 2 * _NGROUPS * _DSTATE
_PROJECTED = _INTERMEDIATE + _CONV_DIM + _NHEADS
_BATCH = 3
_WINDOW = 8
_SLOTS = 2 * _LAYERS * _BATCH


def _build_graph(device: DeviceRef) -> Graph:
    conv_pool = BufferType(
        DType.bfloat16, [_SLOTS, _CONV_DIM, _WIDTH - 1], device=device
    )
    ssm_pool = BufferType(
        DType.float32, [_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE], device=device
    )
    rows = TensorType(DType.uint32, [_LAYERS, _BATCH], device=device)
    rows_total = _BATCH * _WINDOW
    with Graph(
        "mamba2_replay_fold",
        input_types=[
            conv_pool,
            ssm_pool,
            conv_pool,
            ssm_pool,
            rows,
            rows,
            TensorType(DType.int32, [2 * _BATCH], device=device),
            TensorType(DType.int64, ["kept"], device=device),
            TensorType(DType.int32, [_BATCH + 1], device=device),
            *[
                TensorType(DType.bfloat16, [rows_total, _PROJECTED], device)
                for _ in range(_LAYERS)
            ],
            *[
                TensorType(DType.bfloat16, [rows_total, _CONV_DIM], device)
                for _ in range(_LAYERS)
            ],
            *[
                TensorType(DType.bfloat16, [_NHEADS], device)
                for _ in range(2 * _LAYERS)
            ],
            TensorType(DType.bfloat16, [_CONV_DIM, _WIDTH], device),
        ],
    ) as graph:
        (
            fold_conv,
            fold_ssm,
            ref_conv,
            ref_ssm,
            conv_rows,
            ssm_rows,
            bounds,
            kept_rows,
            offsets,
            *rest,
        ) = graph.inputs
        projected = [v.tensor for v in rest[:_LAYERS]]
        convolved = [v.tensor for v in rest[_LAYERS : 2 * _LAYERS]]
        A = [v.tensor for v in rest[2 * _LAYERS : 3 * _LAYERS]]
        dt_bias = [v.tensor for v in rest[3 * _LAYERS : 4 * _LAYERS]]
        conv_weight = rest[4 * _LAYERS].tensor

        mamba2_replay_fold(
            fold_conv.buffer,
            fold_ssm.buffer,
            conv_rows.tensor,
            ssm_rows.tensor,
            bounds.tensor,
            projected,
            convolved,
            A,
            dt_bias,
        )

        group_dim = _NGROUPS * _DSTATE
        always = ops.constant(
            np.ones(_BATCH, dtype=np.bool_), DType.bool, device=device
        )
        # No conv bias and no skip term: neither touches a state.
        empty = ops.constant(
            np.zeros(0, dtype=np.float32), DType.float32, device
        ).cast(DType.bfloat16)
        for layer in range(_LAYERS):
            accepted = ops.gather(projected[layer], kept_rows.tensor, axis=0)
            conv_input = accepted[:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM]
            dt = accepted[:, _INTERMEDIATE + _CONV_DIM :]
            x, B, C = ops.split(
                ops.gather(convolved[layer], kept_rows.tensor, axis=0),
                [_INTERMEDIATE, group_dim, group_dim],
                axis=1,
            )
            causal_conv1d_varlen_fwd(
                conv_input,
                conv_weight,
                empty,
                ref_conv.buffer,
                offsets.tensor,
                conv_rows.tensor[layer],
                always,
                channels_last=True,
            )
            mamba2_ssd_chunk_scan_varlen_fwd_inplace(
                x=ops.reshape(x, [-1, _NHEADS, _HEAD_DIM]),
                dt=dt,
                A=A[layer],
                B=ops.reshape(B, [-1, _NGROUPS, _DSTATE]),
                C=ops.reshape(C, [-1, _NGROUPS, _DSTATE]),
                D=empty,
                dt_bias=dt_bias[layer],
                ssm_pool=ref_ssm.buffer,
                query_start_loc=offsets.tensor,
                has_initial_state=always,
                cache_indices=ssm_rows.tensor[layer],
            )
        graph.output()
    return graph


def _build_fold_graph(device: DeviceRef) -> Graph:
    """Builds the fold alone, with the batch left symbolic."""
    rows = TensorType(DType.uint32, [_LAYERS, "batch"], device=device)
    rows_total = _BATCH * _WINDOW
    with Graph(
        "mamba2_replay_fold_only",
        input_types=[
            BufferType(
                DType.bfloat16, [_SLOTS, _CONV_DIM, _WIDTH - 1], device=device
            ),
            BufferType(
                DType.float32,
                [_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE],
                device=device,
            ),
            rows,
            rows,
            TensorType(DType.int32, ["bounds"], device=device),
            *[
                TensorType(DType.bfloat16, [rows_total, _PROJECTED], device)
                for _ in range(_LAYERS)
            ],
            *[
                TensorType(DType.bfloat16, [rows_total, _CONV_DIM], device)
                for _ in range(_LAYERS)
            ],
            *[
                TensorType(DType.bfloat16, [_NHEADS], device)
                for _ in range(2 * _LAYERS)
            ],
        ],
    ) as graph:
        conv_pool, ssm_pool, conv_rows, ssm_rows, bounds, *rest = graph.inputs
        layers = [v.tensor for v in rest]
        mamba2_replay_fold(
            conv_pool.buffer,
            ssm_pool.buffer,
            conv_rows.tensor,
            ssm_rows.tensor,
            bounds.tensor,
            layers[:_LAYERS],
            layers[_LAYERS : 2 * _LAYERS],
            layers[2 * _LAYERS : 3 * _LAYERS],
            layers[3 * _LAYERS :],
        )
        graph.output()
    return graph


def _bf16(array: np.ndarray, device: md.Device) -> md.Buffer:
    return md.Buffer.from_dlpack(
        torch.from_numpy(array).to(torch.bfloat16).contiguous()
    ).to(device)


@pytest.fixture(scope="module")
def replay_model(session: InferenceSession) -> Model:
    """Compiles the fold and the per-layer replay once for every case."""
    return session.load(_build_graph(DeviceRef.GPU()))


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize("kept", [(1, 8, 4), (0, 2, 8)])
def test_replay_fold_matches_the_per_layer_replay(
    replay_model: Model, kept: tuple[int, int, int]
) -> None:
    """Every layer's conv window and SSM state match bitwise.

    The prefixes cover one row (shorter than the conv window), the whole
    window, and nothing at all.
    """
    device = replay_model.input_devices[0]
    rng = np.random.default_rng(3)

    conv_pool = rng.standard_normal((_SLOTS, _CONV_DIM, _WIDTH - 1)).astype(
        np.float32
    )
    ssm_pool = rng.standard_normal(
        (_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE)
    ).astype(np.float32)
    # Each layer owns distinct rows, in no particular order.
    slots = rng.permutation(_SLOTS)[: 2 * _LAYERS * _BATCH].astype(np.uint32)
    conv_rows = slots[: _LAYERS * _BATCH].reshape(_LAYERS, _BATCH)
    ssm_rows = slots[_LAYERS * _BATCH :].reshape(_LAYERS, _BATCH)
    bounds = np.asarray(
        [
            row
            for b, n in enumerate(kept)
            for row in (b * _WINDOW, b * _WINDOW + n)
        ],
        dtype=np.int32,
    )
    kept_rows = np.asarray(
        [b * _WINDOW + i for b, n in enumerate(kept) for i in range(n)],
        dtype=np.int64,
    )
    offsets = np.cumsum([0, *kept]).astype(np.int32)
    rows_total = _BATCH * _WINDOW
    projected = [
        rng.standard_normal((rows_total, _PROJECTED)).astype(np.float32)
        for _ in range(_LAYERS)
    ]
    convolved = [
        rng.standard_normal((rows_total, _CONV_DIM)).astype(np.float32)
        for _ in range(_LAYERS)
    ]
    A = [-(rng.random(_NHEADS, dtype=np.float32) + 0.1) for _ in range(_LAYERS)]
    dt_bias = [rng.random(_NHEADS, dtype=np.float32) for _ in range(_LAYERS)]

    pools = [
        _bf16(conv_pool, device),
        md.Buffer.from_numpy(ssm_pool.copy()).to(device),
        _bf16(conv_pool, device),
        md.Buffer.from_numpy(ssm_pool.copy()).to(device),
    ]
    replay_model.execute(
        *pools,
        md.Buffer.from_numpy(conv_rows).to(device),
        md.Buffer.from_numpy(ssm_rows).to(device),
        md.Buffer.from_numpy(bounds).to(device),
        md.Buffer.from_numpy(kept_rows).to(device),
        md.Buffer.from_numpy(offsets).to(device),
        *(_bf16(a, device) for a in (*projected, *convolved, *A, *dt_bias)),
        _bf16(
            rng.standard_normal((_CONV_DIM, _WIDTH)).astype(np.float32), device
        ),
    )
    fold_conv, fold_ssm, ref_conv, ref_ssm = (
        torch.from_dlpack(p).cpu().float().numpy() for p in pools
    )
    np.testing.assert_array_equal(fold_conv, ref_conv)
    np.testing.assert_array_equal(fold_ssm, ref_ssm)
    # The fold has to have changed something, or the test proves nothing.
    assert not np.array_equal(fold_ssm, ssm_pool)


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_replay_fold_launches_past_the_grid_z_limit(
    session: InferenceSession,
) -> None:
    """A batch with more than 65535 layer-request blocks lands the same states.

    Every request but the last three accepts nothing, so those three, on the
    highest block indices, must land what they land in a batch of three.
    """
    model = session.load(_build_fold_graph(DeviceRef.GPU()))
    device = model.input_devices[0]
    rng = np.random.default_rng(5)
    kept = (1, 8, 4)
    idle = 65535 // _LAYERS + 1 - len(kept)

    conv_pool = rng.standard_normal((_SLOTS, _CONV_DIM, _WIDTH - 1)).astype(
        np.float32
    )
    ssm_pool = rng.standard_normal(
        (_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE)
    ).astype(np.float32)
    rows = (
        rng.permutation(_SLOTS)[: 2 * _LAYERS * _BATCH]
        .astype(np.uint32)
        .reshape(2, _LAYERS, _BATCH)
    )
    bounds = np.asarray(
        [
            row
            for b, n in enumerate(kept)
            for row in (b * _WINDOW, b * _WINDOW + n)
        ],
        dtype=np.int32,
    )
    rows_total = _BATCH * _WINDOW
    layers = [
        _bf16(a, device)
        for a in (
            *(
                rng.standard_normal((rows_total, _PROJECTED)).astype(np.float32)
                for _ in range(_LAYERS)
            ),
            *(
                rng.standard_normal((rows_total, _CONV_DIM)).astype(np.float32)
                for _ in range(_LAYERS)
            ),
            *(
                -(rng.random(_NHEADS, dtype=np.float32) + 0.1)
                for _ in range(_LAYERS)
            ),
            *(rng.random(_NHEADS, dtype=np.float32) for _ in range(_LAYERS)),
        )
    ]

    def fold(leading: int) -> list[np.ndarray]:
        pools = [
            _bf16(conv_pool, device),
            md.Buffer.from_numpy(ssm_pool.copy()).to(device),
        ]
        padded_rows = np.concatenate(
            [np.zeros((2, _LAYERS, leading), dtype=np.uint32), rows], axis=2
        )
        model.execute(
            *pools,
            md.Buffer.from_numpy(padded_rows[0].copy()).to(device),
            md.Buffer.from_numpy(padded_rows[1].copy()).to(device),
            md.Buffer.from_numpy(
                np.concatenate([np.zeros(2 * leading, np.int32), bounds])
            ).to(device),
            *layers,
        )
        return [torch.from_dlpack(p).cpu().float().numpy() for p in pools]

    small_conv, small_ssm = fold(0)
    large_conv, large_ssm = fold(idle)
    np.testing.assert_array_equal(large_conv, small_conv)
    np.testing.assert_array_equal(large_ssm, small_ssm)
    assert not np.array_equal(small_ssm, ssm_pool)
