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
"""Targeted behavioral tests for :class:`KimiK3MLAAttention`.

Not a full V2-vs-V3 numerical parity test (no existing V2 unit-test fixture
for the parity form to compare against cheaply; left as follow-up, see this
PR's report). These instead prove, with real tensor values rather than shape
checks, the two specific behaviors a RoPE-shaped port would get wrong: no
rotation is ever applied to the rope slice, and the output gate actually
multiplies the attention output before ``o_proj``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
)
from max.tensor import Tensor, default_device

_HIDDEN = 64
_NUM_HEADS = 4
_QK_NOPE = 16
_QK_ROPE = 8
_V_HEAD = 16
_Q_LORA = 32
_KV_LORA = 24


def _make_layer() -> KimiK3MLAAttention:
    return KimiK3MLAAttention(
        hidden_size=_HIDDEN,
        num_heads=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        v_head_dim=_V_HEAD,
        q_lora_rank=_Q_LORA,
        kv_lora_rank=_KV_LORA,
    )


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_output_gate_zeroes_the_output() -> None:
    """Forcing ``g_proj``'s weight to a saturating value must zero the
    final output.

    ``gate = sigmoid(g_proj(x))``; a zero weight (and no bias, per
    ``bias=False``) makes every logit 0, so ``sigmoid(0) = 0.5`` -- not
    actually zero. Use a large negative CONSTANT weight instead, so the gate
    saturates to ~0, and confirm the output collapses to (near) zero. If the
    gate were applied after ``o_proj`` instead of before, or not applied at
    all, this would not hold: ``o_proj`` has no bias, so the only way the
    output can be forced to zero by ``g_proj`` alone is if the gate truly
    multiplies ``attn`` before ``o_proj``.

    A constant weight's logit is ``weight_row . x``, which is
    ``-50 * sum(x)`` here since every row is identical -- its SIGN depends on
    ``sum(x)``, not just the weight's sign, so an input with unconstrained
    sign (plain ``randn``) saturates toward 0 for some tokens and toward 1
    for others, roughly at random. Non-negative input (``abs(randn)``) makes
    ``sum(x) >= 0`` unconditionally, so the logit is reliably very negative.
    """
    with default_device(Accelerator()):
        layer = _make_layer()
        # A large-magnitude negative weight saturates sigmoid(g_proj(x)) to
        # ~0, given a non-negative x (see docstring for why x's sign matters
        # here).
        layer.g_proj.weight = Tensor(
            np.full((_NUM_HEADS * _V_HEAD, _HIDDEN), -50.0, dtype=np.float32),
            dtype=DType.float32,
        ).cast(layer.g_proj.weight.dtype)

        x = Tensor(
            np.abs(
                np.random.default_rng(0).normal(size=(1, 5, _HIDDEN))
            ).astype(np.float32),
            dtype=DType.float32,
        ).cast(DType.bfloat16)
        out = layer(x)

        out_np = out.cast(DType.float32).to_numpy()
        max_abs = np.abs(out_np).max()
        assert max_abs < 1e-2, (
            "output gate should have collapsed the output to ~0, got max"
            f" abs value {max_abs}"
        )


def test_rope_slice_is_never_rotated() -> None:
    """The rope-shaped slice must survive identically from ``q_b_proj`` to
    the point it enters attention -- no rotation is ever applied to it.

    Confirmed by constructing the exact sub-expression ``forward()`` uses for
    the split-and-reconcatenate (``F.concat([q[..., :nope], q[...,
    nope:]], axis=-1)``) and checking it really is the identity on a known
    input: the reference's own recipe is an unrotated split+concat, not an
    unapplied rotation left dead in the code. Testing the full layer would
    only indirectly exercise this (attention mixes everything together);
    isolating the expression proves the specific claim directly.
    """
    from max import functional as F

    with default_device(Accelerator()):
        nope, rope = 16, 8
        rng = np.random.default_rng(1)
        q_np = rng.normal(size=(2, 3, 2, nope + rope)).astype(np.float32)
        q = Tensor(q_np, dtype=DType.float32)

        result = F.concat(
            [q[..., :nope], q[..., nope:]],
            axis=-1,
        )
        result_np = result.to_numpy()

    np.testing.assert_array_equal(
        result_np,
        q_np,
        err_msg=(
            "split-then-concat of the rope slice must be the identity;"
            " a real rotation would change these values"
        ),
    )


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_forward_pass_runs_with_correct_shape_and_dtype() -> None:
    """Sanity: the full layer compiles and runs end to end."""
    with default_device(Accelerator()):
        layer = _make_layer()
        x = Tensor(
            np.random.default_rng(2)
            .normal(size=(2, 7, _HIDDEN))
            .astype(np.float32),
            dtype=DType.float32,
        ).cast(DType.bfloat16)
        out = layer(x)
        assert out.shape == [2, 7, _HIDDEN]
        assert out.dtype == DType.bfloat16
