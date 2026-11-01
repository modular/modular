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
"""GPU execution tests (TP1) for the V4.1 mHC gate path and stream chains.

The chain tests execute the exact shifted schedule the trunk runs
(initial identity -> default (fresh-``pre``) site -> identity site ->
final collapse) on one compiled device, plus the production-shaped chain
whose sublayer outputs read the collapses. The TP2 counterparts live in
``test_mhc_tp2_gpu.py``.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import mhc_pre
from mhc_chain_reference import (
    HC,
    HC_EPS,
    HIDDEN,
    NORM_EPS,
    SINKHORN_ITERS,
    collapse_oracle,
    gates_oracle,
    proj_oracle,
    round_bf16,
    run_chain,
    run_prod_chain,
    site_params,
    to_bf16_device,
    to_f32_device,
)


def test_gates_match_the_kernel_recipe() -> None:
    """``mhc_pre``'s coefficients at the production stream width 5120."""
    hidden = 5120
    rows = 9
    rng = np.random.default_rng(40)
    stream = round_bf16(
        (rng.standard_normal((rows, HC, hidden)) * 0.3).astype(np.float32)
    )
    fn, base, scale = site_params(rng, hidden)

    post, comb, pre, collapsed = mhc_pre(
        to_bf16_device(stream),
        to_f32_device(fn),
        to_f32_device(scale),
        to_f32_device(base),
        HC,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    assert post.dtype == DType.float32
    assert pre.dtype == DType.float32
    assert comb.dtype == DType.float32

    proj = proj_oracle(stream, fn, hidden)
    want_pre, want_post, want_comb = gates_oracle(
        proj, base, scale, SINKHORN_ITERS
    )
    # The 5120-wide fp32 projection may run at TF32 on GPU, which sets
    # the accuracy floor (measured: 9.1e-4 relative on this fixture).
    np.testing.assert_allclose(pre.to_numpy(), want_pre, rtol=2e-3, atol=1e-3)
    np.testing.assert_allclose(post.to_numpy(), want_post, rtol=2e-3, atol=1e-3)
    np.testing.assert_allclose(
        comb.to_numpy().reshape(rows, HC * HC), want_comb, rtol=2e-3, atol=1e-3
    )
    # Default pre_mix: the site's own fresh pre collapses the stream.
    want_collapsed = round_bf16(collapse_oracle(stream, want_pre))
    np.testing.assert_allclose(
        collapsed.cast(DType.float32).to_numpy(),
        want_collapsed,
        rtol=1e-2,
        atol=1e-2,
    )


def test_shifted_two_site_chain_single_device() -> None:
    """initial -> default pre -> identity pre -> collapse, compiled TP1."""
    got, want = run_chain(None, HIDDEN)
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)


def test_production_chain_collapses_are_live_tp1() -> None:
    """Each sublayer reads its collapse: compiled TP1, no dead collapses."""
    got, want = run_prod_chain(None, HIDDEN)
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)
