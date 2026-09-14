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
"""Layout-equivalence test for ``causal_conv1d_varlen_fwd`` ``channels_last``.

The builtin ``causal_conv1d_varlen_fwd`` (Nemotron-H mamba conv) historically
required channels-first ``(dim, total_seqlen)`` tensors, forcing the model to
materialize a transpose on each side of the op. The ``channels_last``
parameter lets the op consume/produce tokens-major ``(total_seqlen, dim)``
directly; the kernels index through runtime strides, so both layouts must run
the exact same per-element arithmetic.

This test builds one graph containing BOTH paths on identical inputs:

* channels-first arm: ``transpose -> op(channels_last=False) -> transpose``
  — the legacy contract;
* channels-last arm: ``op(channels_last=True)`` on the tokens-major tensor.

Each pair runs twice, with ``use_residual`` off and on: the residual reads
``x`` through the same strided accessor the layout parameter steers.

The test asserts bitwise-identical outputs and conv-state pools for a ragged
prefill step followed by a state-carrying decode step (one token per
sequence, ``has_initial_state=True``), reusing the same compiled model via a
symbolic ``total_seqlen`` dimension.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
    ops,
)

# Nemotron-H conv kernel width (widths 1-4 are compiled in the builtin).
_KERNEL_SIZE = 4
# Deliberately NOT a multiple of the kernel's BLOCK_DIM (128) so the
# out-of-range channel guard is exercised in both layouts.
_DIM = 192
# The graph input carries extra leading/trailing columns and the conv input
# is a nonzero-offset strided slice of it — mirroring the production caller,
# where the conv consumes a strided `ops.split` view of the fused in_proj
# output.
_COL_OFFSET = 40
_WIDE_DIM = _COL_OFFSET + _DIM + 24
_MAX_SLOTS = 4
_STATE_LEN = _KERNEL_SIZE - 1
_BATCH = 2
_SLOTS = [1, 3]

# The four arms, in graph-output and conv-state-pool order: each layout with
# and without the fused residual add, as (channels_last, use_residual).
_ARMS = [(False, False), (True, False), (False, True), (True, True)]
_CF, _CL, _CF_RES, _CL_RES = range(len(_ARMS))
# Conv-state pools are graph inputs 3..6; the ragged metadata follows.
_FIRST_POOL_INPUT = 3


def _conv_arm(
    gpu: DeviceRef,
    x_cl: TensorValue,
    operands: list[TensorValue | BufferValue],
    *,
    channels_last: bool,
    use_residual: bool,
) -> TensorValue:
    """One conv arm, returning tokens-major ``[N, dim]`` either way.

    Every kernel parameter is passed explicitly: the op does not inherit
    the defaults the Mojo struct declares.
    """
    x = ops.transpose(x_cl, 0, 1) if not channels_last else x_cl
    n = x_cl.shape[0]
    shape: list[DimLike] = [n, _DIM] if channels_last else [_DIM, n]
    out = ops.inplace_custom(
        "causal_conv1d_varlen_fwd",
        gpu,
        [x, *operands],
        [TensorType(DType.float32, shape, device=gpu)],
        parameters={
            "activation": "silu",
            "channels_last": channels_last,
            "use_residual": use_residual,
        },
    )[0].tensor
    return out if channels_last else ops.transpose(out, 0, 1)


def _build_dual_layout_graph(gpu: DeviceRef) -> Graph:
    """One graph computing the conv through all four arms.

    Each arm needs its own conv-state pool, since every arm mutates its pool
    in place.
    """
    pool_type = BufferType(
        DType.float32, [_MAX_SLOTS, _DIM, _STATE_LEN], device=gpu
    )
    with Graph(
        "causal_conv1d_channels_last_equivalence",
        input_types=[
            TensorType(DType.float32, ["total_seqlen", _WIDE_DIM], device=gpu),
            TensorType(DType.float32, [_DIM, _KERNEL_SIZE], device=gpu),
            TensorType(DType.float32, [_DIM], device=gpu),
            *([pool_type] * len(_ARMS)),
            TensorType(DType.int32, [_BATCH + 1], device=gpu),
            TensorType(DType.int32, [_BATCH], device=gpu),
            TensorType(DType.bool, [_BATCH], device=gpu),
        ],
    ) as graph:
        x_wide = graph.inputs[0].tensor  # [N, wide_dim] tokens-major
        # Nonzero-offset strided slice, as the production caller feeds the op.
        x_cl = ops.slice_tensor(
            x_wide, [slice(None), slice(_COL_OFFSET, _COL_OFFSET + _DIM)]
        )  # [N, dim]
        weight = graph.inputs[1].tensor
        bias = graph.inputs[2].tensor
        meta = _FIRST_POOL_INPUT + len(_ARMS)
        qsl = graph.inputs[meta].tensor
        cache_indices = graph.inputs[meta + 1].tensor
        has_initial_state = graph.inputs[meta + 2].tensor

        graph.output(
            *[
                _conv_arm(
                    gpu,
                    x_cl,
                    [
                        weight,
                        bias,
                        graph.inputs[_FIRST_POOL_INPUT + arm].buffer,
                        qsl,
                        cache_indices,
                        has_initial_state,
                    ],
                    channels_last=channels_last,
                    use_residual=use_residual,
                )
                for arm, (channels_last, use_residual) in enumerate(_ARMS)
            ]
        )
    return graph


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_causal_conv1d_channels_last_matches_channels_first(
    session: InferenceSession,
) -> None:
    """channels_last output/state must be bitwise-equal to channels-first.

    Holds with the fused residual add on as well as off.
    """
    gpu = DeviceRef.GPU()
    model = session.load(_build_dual_layout_graph(gpu))
    gpu_device = model.input_devices[0]

    rng = np.random.default_rng(1234)
    weight_np = rng.standard_normal((_DIM, _KERNEL_SIZE)).astype(np.float32)
    bias_np = rng.standard_normal((_DIM,)).astype(np.float32)
    pool_initial_np = rng.standard_normal(
        (_MAX_SLOTS, _DIM, _STATE_LEN)
    ).astype(np.float32)
    slot_idx_np = np.asarray(_SLOTS, dtype=np.int32)
    untouched_slots = [s for s in range(_MAX_SLOTS) if s not in _SLOTS]

    # One pool per arm, in `_ARMS` order, so no two arms write the same slots.
    pool_bufs = [
        md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
        for _ in _ARMS
    ]
    weight_buf = md.Buffer.from_numpy(weight_np).to(gpu_device)
    bias_buf = md.Buffer.from_numpy(bias_np).to(gpu_device)
    slot_buf = md.Buffer.from_numpy(slot_idx_np).to(gpu_device)

    def _run(
        x_np: np.ndarray, offsets: list[int], has_init: bool
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        outputs = model.execute(
            md.Buffer.from_numpy(x_np).to(gpu_device),
            weight_buf,
            bias_buf,
            *pool_bufs,
            md.Buffer.from_numpy(np.asarray(offsets, dtype=np.int32)).to(
                gpu_device
            ),
            slot_buf,
            md.Buffer.from_numpy(np.asarray([has_init] * _BATCH)).to(
                gpu_device
            ),
        )
        outs = [torch.from_dlpack(o).cpu().numpy() for o in outputs]
        pools = [torch.from_dlpack(p).cpu().numpy() for p in pool_bufs]
        return outs, pools

    def _check_step(outs: list[np.ndarray], pools: list[np.ndarray]) -> None:
        """Each layout pair must agree bitwise, with and without residual."""
        np.testing.assert_array_equal(outs[_CL], outs[_CF])
        np.testing.assert_array_equal(pools[_CL], pools[_CF])
        np.testing.assert_array_equal(outs[_CL_RES], outs[_CF_RES])
        np.testing.assert_array_equal(pools[_CL_RES], pools[_CF_RES])
        # The residual arms must not be silently computing the plain conv;
        # the conv state is the x window either way, so only outputs differ.
        assert not np.array_equal(outs[_CF_RES], outs[_CF]), (
            "use_residual=True produced the same output as use_residual=False"
        )
        np.testing.assert_array_equal(pools[_CF_RES], pools[_CF])

    # Ragged prefill: two fresh sequences of lengths 7 and 5.
    prefill_len = 12
    x_prefill = rng.standard_normal((prefill_len, _WIDE_DIM)).astype(np.float32)
    outs, pools = _run(x_prefill, [0, 7, prefill_len], has_init=False)
    _check_step(outs, pools)
    for s in _SLOTS:
        assert not np.array_equal(pools[_CF][s], pool_initial_np[s]), (
            f"conv-state slot {s} should have been mutated by prefill"
        )
    for s in untouched_slots:
        for pool in pools:
            np.testing.assert_array_equal(pool[s], pool_initial_np[s])

    # Decode: one new token per sequence, carrying the stored conv state.
    x_decode = rng.standard_normal((_BATCH, _WIDE_DIM)).astype(np.float32)
    outs, pools = _run(x_decode, [0, 1, _BATCH], has_init=True)
    _check_step(outs, pools)
