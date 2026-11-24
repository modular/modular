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
"""Verifies TP=2 matches TP=1 for the fused attention-residual mixture."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.attn_residual import (
    PendingAttnResStream,
    kimi_k3_attn_res_mix_norm,
    kimi_k3_attn_res_mix_norm_delta,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


@pytest.mark.compile_on_gpu("attn_res_mix_norm has no custom.declare")
def test_attn_res_mix_norm_tp2_matches_tp1() -> None:
    hidden, total_seq_len, num_candidates = 32, 5, 3
    rng = np.random.default_rng(0)

    prefix_sum_np = rng.standard_normal((total_seq_len, hidden)).astype(
        np.float32
    )
    block_residual_np = [
        rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
        for _ in range(num_candidates - 1)
    ]
    proj_weight_np = rng.standard_normal((1, hidden)).astype(np.float32)
    res_norm_weight_np = rng.standard_normal(hidden).astype(np.float32)
    out_norm_weight_np = rng.standard_normal(hidden).astype(np.float32)

    with default_device(Accelerator(0)):
        out1 = kimi_k3_attn_res_mix_norm(
            Tensor(prefix_sum_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
            Tensor(proj_weight_np, dtype=DType.float32),
            Tensor(res_norm_weight_np, dtype=DType.float32),
            Tensor(out_norm_weight_np, dtype=DType.float32),
        ).to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        out2 = kimi_k3_attn_res_mix_norm(
            Tensor(prefix_sum_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
            Tensor(proj_weight_np, dtype=DType.float32),
            Tensor(res_norm_weight_np, dtype=DType.float32),
            Tensor(out_norm_weight_np, dtype=DType.float32),
        ).to_numpy()

    err = _rel_error(out2, out1)
    assert err < 1e-4, (
        f"attn_res_mix_norm TP=2 vs TP=1 diverges: rel_error={err}"
    )


@pytest.mark.compile_on_gpu("attn_res_mix_norm_delta has no custom.declare")
def test_attn_res_mix_norm_delta_tp2_matches_tp1() -> None:
    """The folded mixture agrees across TP=2 and TP=1 on both outputs --
    the normed value and the materialized sum the next layer consumes."""
    hidden, total_seq_len, num_candidates = 32, 5, 3
    rng = np.random.default_rng(1)

    prefix_np = rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
    block_residual_np = [
        rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
        for _ in range(num_candidates - 1)
    ]
    delta_np = rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
    proj_weight_np = rng.standard_normal((1, hidden)).astype(np.float32)
    res_norm_weight_np = rng.standard_normal(hidden).astype(np.float32)
    out_norm_weight_np = rng.standard_normal(hidden).astype(np.float32)

    def run(
        stream: PendingAttnResStream, stack: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        return kimi_k3_attn_res_mix_norm_delta(
            stream.prefix,
            stream.delta,
            stack,
            Tensor(proj_weight_np, dtype=DType.float32),
            Tensor(res_norm_weight_np, dtype=DType.float32),
            Tensor(out_norm_weight_np, dtype=DType.float32),
        )

    with default_device(Accelerator(0)):
        normed1, summed1 = run(
            PendingAttnResStream(
                Tensor(prefix_np, dtype=DType.float32),
                Tensor(delta_np, dtype=DType.float32),
            ),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
        )
        normed1_np = normed1.to_numpy()
        summed1_np = summed1.to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        normed2, summed2 = run(
            PendingAttnResStream(
                Tensor(prefix_np, dtype=DType.float32),
                Tensor(delta_np, dtype=DType.float32),
            ),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
        )
        normed2_np = normed2.to_numpy()
        summed2_np = summed2.to_numpy()

    err_normed = _rel_error(normed2_np, normed1_np)
    err_sum = _rel_error(summed2_np, summed1_np)
    assert err_normed < 1e-4, (
        f"attn_res_mix_norm_delta normed TP=2 vs TP=1 diverges:"
        f" rel_error={err_normed}"
    )
    assert err_sum < 1e-4, (
        f"attn_res_mix_norm_delta sum TP=2 vs TP=1 diverges:"
        f" rel_error={err_sum}"
    )
