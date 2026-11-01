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
"""Numerical verification for :func:`kimi_situ_and_mul`,
:func:`kimi_situ_and_mul_quantize_mxfp8`, and :class:`KimiK3DenseMLP`.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, accelerator_api, accelerator_count
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
    kimi_situ_and_mul,
    kimi_situ_and_mul_quantize_mxfp8,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    quantize_mxfp8,
)
from max.tensor import Tensor, default_device
from test_common.graph_utils import is_gfx950


def _numpy_situ_and_mul(
    gate_up: np.ndarray, beta: float, linear_beta: float | None
) -> np.ndarray:
    width = gate_up.shape[-1] // 2
    gate = gate_up[..., :width].astype(np.float64)
    up = gate_up[..., width:].astype(np.float64)
    situ = beta * np.tanh(gate / beta) * (1.0 / (1.0 + np.exp(-gate)))
    if linear_beta is not None:
        up = linear_beta * np.tanh(up / linear_beta)
    return situ * up


def test_kimi_situ_and_mul_matches_independent_reference() -> None:
    rng = np.random.default_rng(0)
    beta, linear_beta = 1.5, 1.0
    seq_len, width = 5, 16
    gate_up_np = rng.standard_normal((seq_len, 2 * width)).astype(np.float32)
    expected = _numpy_situ_and_mul(gate_up_np, beta, linear_beta)

    with default_device(Accelerator()):
        gate_up = Tensor(gate_up_np, dtype=DType.float32).cast(DType.bfloat16)
        out = kimi_situ_and_mul(gate_up, beta, linear_beta, width)
        out_np = out.cast(DType.float32).to_numpy().astype(np.float64)

    assert out.dtype == DType.bfloat16
    rel_error = np.max(np.abs(out_np - expected)) / max(
        np.max(np.abs(expected)), 1e-12
    )
    assert rel_error < 1e-2, (
        f"kimi_situ_and_mul diverges: rel_error={rel_error}"
    )


def test_kimi_situ_and_mul_passes_up_through_unchanged_when_linear_beta_none() -> (
    None
):
    """``linear_beta=None`` is the up half's identity, not a disguised zero."""
    rng = np.random.default_rng(1)
    beta = 2.0
    seq_len, width = 4, 8
    gate_up_np = rng.standard_normal((seq_len, 2 * width)).astype(np.float32)
    expected = _numpy_situ_and_mul(gate_up_np, beta, None)

    with default_device(Accelerator()):
        gate_up = Tensor(gate_up_np, dtype=DType.float32)
        out_np = (
            kimi_situ_and_mul(gate_up, beta, None, width)
            .to_numpy()
            .astype(np.float64)
        )

    rel_error = np.max(np.abs(out_np - expected)) / max(
        np.max(np.abs(expected)), 1e-12
    )
    # kimi_situ_and_mul's tanh/sigmoid run in float32 on the GPU; different
    # vendors' transcendental-function implementations diverge from a
    # float64 CPU reference by about this much (same reasoning as
    # test_moe_router.py's rel_error tolerance -- confirmed on CI hardware
    # (NVIDIA B200) this port hadn't been tested against before).
    assert rel_error < 1e-3, (
        f"linear_beta=None path diverges: rel_error={rel_error}"
    )


@pytest.mark.compile_on_gpu(
    "mo.quantize.dynamic.block.scaled and the fused swiglu quantize are"
    " custom ops the interpreter can't bind"
)
def test_kimi_situ_and_mul_quantize_mxfp8_matches_chain_bytes() -> None:
    """The fused SiTU+quantize wrapper vs the chain it replaces, byte-wise.

    ``kimi_situ_and_mul`` + ``quantize_mxfp8`` (what the routed experts ran)
    and the fused kernel quantize the same bf16-rounded SiTU, but their fp32
    transcendentals differ by an ulp on some inputs, which can flip a bf16
    rounding at a block max and shift that block's scale by one E8M0 step.
    The mismatch count is therefore gated small, not zero: a real divergence
    (a swapped gate/up, a launch geometry on 2N, a missing scale column)
    mismatches whole blocks, orders of magnitude past the gate.
    """
    if (
        accelerator_count() == 0
        or accelerator_api() != "hip"
        or not is_gfx950()
    ):
        pytest.skip("needs an AMD gfx950 GPU")
    beta, linear_beta = 4.0, 25.0
    rng = np.random.default_rng(7)
    max_scale_diffs = 4
    max_data_diffs = 128
    for m, width in [(64, 384), (8, 3072)]:
        # Signed data past both saturations: |gate| > 9 * beta drives tanh
        # to +-1, |up| > linear_beta drives the soft saturation.
        gate_up_np = (rng.standard_normal((m, 2 * width)) * 2).astype(
            np.float32
        )
        gate_up_np[:, 0:width:9] = 40.0
        gate_up_np[:, width::11] = 100.0

        with default_device(Accelerator()):
            gate_up = Tensor(gate_up_np, dtype=DType.float32).cast(
                DType.bfloat16
            )
            chain_q, chain_scales = quantize_mxfp8(
                kimi_situ_and_mul(gate_up, beta, linear_beta, width)
            )
            fused_q, fused_scales = kimi_situ_and_mul_quantize_mxfp8(
                gate_up, beta, linear_beta
            )
            neg_q, neg_scales = kimi_situ_and_mul_quantize_mxfp8(
                gate_up, beta, -linear_beta
            )
            chain_q_bytes = chain_q.storage.view(DType.uint8).to_numpy()
            chain_scales_bytes = chain_scales.storage.view(
                DType.uint8
            ).to_numpy()
            fused_q_bytes = fused_q.storage.view(DType.uint8).to_numpy()
            fused_scales_bytes = fused_scales.storage.view(
                DType.uint8
            ).to_numpy()

        # beta * tanh(x / beta) is even in beta, so a negative linear_beta
        # must build and match its positive twin.
        np.testing.assert_array_equal(
            neg_q.storage.view(DType.uint8).to_numpy(), fused_q_bytes
        )
        np.testing.assert_array_equal(
            neg_scales.storage.view(DType.uint8).to_numpy(),
            fused_scales_bytes,
        )

        scale_diffs = int(
            np.count_nonzero(fused_scales_bytes != chain_scales_bytes)
        )
        data_diffs = int(np.count_nonzero(fused_q_bytes != chain_q_bytes))
        print(
            f"M={m} gate_up={2 * width}: fused vs chain:"
            f" {data_diffs} of {fused_q_bytes.size} data bytes,"
            f" {scale_diffs} of {fused_scales_bytes.size} scales differ"
        )
        assert scale_diffs <= max_scale_diffs, (
            f"M={m}: {scale_diffs} scales differ"
        )
        assert data_diffs <= max_data_diffs, (
            f"M={m}: {data_diffs} data bytes differ"
        )


def test_dense_mlp_matches_independent_reference() -> None:
    """``down_proj(situ_and_mul(gate_up_proj(x)))`` end to end, real weights."""
    rng = np.random.default_rng(2)
    hidden, intermediate = 16, 20
    beta, linear_beta = 1.2, 0.8
    seq_len = 6

    x_np = rng.standard_normal((seq_len, hidden)).astype(np.float32)
    gate_up_weight_np = (
        rng.standard_normal((2 * intermediate, hidden)) / np.sqrt(hidden)
    ).astype(np.float32)
    down_weight_np = (
        rng.standard_normal((hidden, intermediate)) / np.sqrt(intermediate)
    ).astype(np.float32)

    gate_up64 = x_np.astype(np.float64) @ gate_up_weight_np.astype(np.float64).T
    activated64 = _numpy_situ_and_mul(gate_up64, beta, linear_beta)
    expected = activated64 @ down_weight_np.astype(np.float64).T

    with default_device(Accelerator()):
        layer = KimiK3DenseMLP(
            hidden_size=hidden,
            intermediate_size=intermediate,
            beta=beta,
            linear_beta=linear_beta,
        )
        layer.gate_up_proj.weight = Tensor(
            gate_up_weight_np, dtype=DType.float32
        )
        layer.down_proj.weight = Tensor(down_weight_np, dtype=DType.float32)

        x = Tensor(x_np, dtype=DType.float32)
        out_np = layer(x).to_numpy().astype(np.float64)

    assert out_np.shape == (seq_len, hidden)
    rel_error = np.max(np.abs(out_np - expected)) / max(
        np.max(np.abs(expected)), 1e-12
    )
    # Same GPU-vendor transcendental-function precision gap as the sibling
    # test above -- observed on CI's NVIDIA B200 (rel_error ~2e-4), not a
    # logic bug: this exact test has passed reliably and repeatedly on AMD
    # hardware throughout this port's development.
    assert rel_error < 1e-3, f"KimiK3DenseMLP diverges: rel_error={rel_error}"
