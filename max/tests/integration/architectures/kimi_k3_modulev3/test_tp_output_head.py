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
"""Verifies TP=2 matches TP=1 for the output head's vocab-parallel lm_head."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.sharding import DeviceMesh, Replicated
from max.experimental.tensor import Tensor, default_device
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.kimi_k3_modulev3.layers.output_head import (
    kimi_k3_output_head,
    tensor_parallel_lm_head_weight,
)


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


@pytest.mark.compile_on_gpu("the interpreter has no gather with uint32 indices")
def test_output_head_tp2_matches_tp1() -> None:
    hidden, vocab = 16, 32
    rng = np.random.default_rng(0)

    input_row_offsets_np = np.array([0, 3, 8], dtype=np.uint32)
    total_seq_len = int(input_row_offsets_np[-1])

    h_np = rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
    norm_weight_np = rng.standard_normal(hidden).astype(np.float32)
    lm_head_np = (
        rng.standard_normal((vocab, hidden)) / np.sqrt(hidden)
    ).astype(np.float32)
    return_n_logits_np = np.array([1], dtype=np.int64)

    with default_device(Accelerator(0)):
        h1 = Tensor(h_np, dtype=DType.float32)
        norm_weight1 = Tensor(norm_weight_np, dtype=DType.float32)
        lm_head_weight1 = Tensor(lm_head_np, dtype=DType.float32)
        input_row_offsets1 = Tensor(input_row_offsets_np, dtype=DType.uint32)
        return_n_logits1 = Tensor(return_n_logits_np, dtype=DType.int64)

        (last_logits1,) = kimi_k3_output_head(
            h1,
            input_row_offsets1,
            return_n_logits1,
            norm_weight1,
            lm_head_weight1,
            return_logits=ReturnLogits.LAST_TOKEN,
        )
        out1 = last_logits1.to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        h2 = Tensor(h_np, dtype=DType.float32)
        norm_weight2 = Tensor(norm_weight_np, dtype=DType.float32)
        lm_head_weight2 = tensor_parallel_lm_head_weight(
            Tensor(lm_head_np, dtype=DType.float32), mesh
        )
        input_row_offsets2 = Tensor(input_row_offsets_np, dtype=DType.uint32)
        return_n_logits2 = Tensor(return_n_logits_np, dtype=DType.int64)

        (last_logits2,) = kimi_k3_output_head(
            h2,
            input_row_offsets2,
            return_n_logits2,
            norm_weight2,
            lm_head_weight2,
            return_logits=ReturnLogits.LAST_TOKEN,
        )
        # `to_numpy` gathers shards itself, so check that the head did.
        assert last_logits2.placements == (Replicated(),)
        out2 = last_logits2.to_numpy()

    err = _rel_error(out2, out1)
    assert err < 1e-4, f"output head TP=2 vs TP=1 diverges: rel_error={err}"
