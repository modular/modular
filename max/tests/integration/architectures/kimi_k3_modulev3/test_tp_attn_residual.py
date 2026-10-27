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
"""Verifies TP=2 matches TP=1 for the attention-residual mixture."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.attn_residual import (
    kimi_k3_attn_res_mix,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


@pytest.mark.compile_on_gpu("attn_res_mix has no custom.declare declaration")
def test_attn_res_mix_tp2_matches_tp1() -> None:
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
    norm_weight_np = rng.standard_normal(hidden).astype(np.float32)

    with default_device(Accelerator(0)):
        out1 = kimi_k3_attn_res_mix(
            Tensor(prefix_sum_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
            Tensor(proj_weight_np, dtype=DType.float32),
            Tensor(norm_weight_np, dtype=DType.float32),
        ).to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        out2 = kimi_k3_attn_res_mix(
            Tensor(prefix_sum_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
            Tensor(proj_weight_np, dtype=DType.float32),
            Tensor(norm_weight_np, dtype=DType.float32),
        ).to_numpy()

    err = _rel_error(out2, out1)
    assert err < 1e-4, f"attn_res_mix TP=2 vs TP=1 diverges: rel_error={err}"
