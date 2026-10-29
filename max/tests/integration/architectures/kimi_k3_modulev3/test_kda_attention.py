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
"""Targeted tests for :class:`KimiK3KDAAttention`.

Not a numerical parity test against V2: the `kda_decode` kernel
L2-normalizes q/k and folds the query scale in internally, so a
hand-derived expected value risks being wrong about the kernel's own
preprocessing rather than the layer's orchestration around it. These
instead prove the layer compiles and runs at both of V2's supported
gate-mode branches, and that the "fail loudly on an unsupported gate bound"
guard V2 has is preserved.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.tensor import Tensor, default_device

_HIDDEN = 64
_NUM_HEADS = 4
# `kda_decode` is only compiled for (key_head_dim, value_head_dim) =
# (32, 32) or (128, 128) -- not an arbitrary test size. 32 matches K3's own
# 0.4B bring-up fixture geometry (`kimi_k3/tests/test_kda_attention_gpu.py`).
_HEAD_DIM = 32
_CONV_KERNEL = 4
_MAX_SLOTS = 2


def _make_layer(gate_lower_bound: float | None) -> KimiK3KDAAttention:
    return KimiK3KDAAttention(
        hidden_size=_HIDDEN,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        conv_kernel_size=_CONV_KERNEL,
        gate_lower_bound=gate_lower_bound,
    )


def _make_pools_and_input(
    total_seq_len: int, batch_size: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    inner = _NUM_HEADS * _HEAD_DIM
    # Float32 regardless of activation dtype -- see V2's
    # `kimi_k3/tests/test_kda_attention_gpu.py`: "the pool is float32 either
    # way."
    conv_pool = Tensor.zeros(
        [_MAX_SLOTS, 3 * inner, _CONV_KERNEL - 1], dtype=DType.float32
    )
    state_pool = Tensor.zeros(
        [_MAX_SLOTS, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM], dtype=DType.float32
    )
    slot_idx = Tensor(
        np.arange(batch_size, dtype=np.uint32), dtype=DType.uint32
    )
    input_row_offsets = Tensor(
        np.array(
            [i * (total_seq_len // batch_size) for i in range(batch_size + 1)],
            dtype=np.uint32,
        ),
        dtype=DType.uint32,
    )
    x = Tensor(
        np.random.default_rng(0)
        .normal(size=(total_seq_len, _HIDDEN))
        .astype(np.float32),
        dtype=DType.float32,
    ).cast(DType.bfloat16)
    return x, conv_pool, state_pool, slot_idx, input_row_offsets


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
@pytest.mark.parametrize("gate_lower_bound", [-5.0, None])
def test_forward_pass_runs_with_correct_shape_and_dtype(
    gate_lower_bound: float | None,
) -> None:
    """Both of V2's gate-mode branches ("safe" and "original") compile and
    run, producing the right shape and dtype -- the same bar V2's own
    production-width shapes test holds at this hardware class (compile and
    run only; that test defers numerics to its separate, smaller fixture)."""
    with default_device(Accelerator()):
        layer = _make_layer(gate_lower_bound)
        x, conv_pool, state_pool, slot_idx, input_row_offsets = (
            _make_pools_and_input(total_seq_len=6, batch_size=2)
        )
        out = layer(
            x, conv_pool, slot_idx, state_pool, slot_idx, input_row_offsets
        )
        assert out.shape == [6, _HIDDEN]
        assert out.dtype == DType.bfloat16


def test_unsupported_gate_lower_bound_raises() -> None:
    """V2 fails loudly rather than silently running the wrong gate when the
    declared bound isn't the -5.0 both kernels hardcode in their safe arm."""
    with pytest.raises(NotImplementedError, match="hardcodes a lower bound"):
        _make_layer(gate_lower_bound=-3.0)
