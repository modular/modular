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
"""GPU test for the fused mHC boundary against the composed chain it replaced.

``mhc_fused_post_pre`` runs main's ``hyper_connection.mega_mhc`` (the write-back, the gate
step, the collapse and the sublayer's RMSNorm in one kernel); the oracle is
the composed ``mhc_post``, ``mhc_pre`` and ``RMSNorm`` on the same inputs.
The gate projection runs in TF32 in both, which sets the gate tolerance.
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    _hc_mean,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import (
    mhc_fused_post_pre,
    mhc_identity_writeback,
    mhc_post,
    mhc_pre,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.mxfp8_linear import (
    MXFP8Linear,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.norm import (
    RMSNorm,
)
from max.tensor import Tensor
from mhc_chain_reference import (
    HC,
    HC_EPS,
    NORM_EPS,
    SINKHORN_ITERS,
    round_bf16,
    site_params,
    to_bf16_device,
    to_f32_device,
)

# The kernel needs a hidden width in whole 1024s; 4096 is V4.1-Flash's.
_HIDDEN = 4096
_ROWS = 11


def _report(name: str, got: np.ndarray, want: np.ndarray) -> None:
    err = np.abs(got - want)
    rel = err / np.maximum(np.abs(want), 1e-2)
    print(f"[receipt] {name}: max abs {err.max():.4g}, max rel {rel.max():.4g}")


def test_fused_boundary_matches_the_composed_chain() -> None:
    rng = np.random.default_rng(50)
    stream = round_bf16(
        (rng.standard_normal((_ROWS, HC, _HIDDEN)) * 0.3).astype(np.float32)
    )
    y = round_bf16(
        (rng.standard_normal((_ROWS, _HIDDEN)) * 0.5).astype(np.float32)
    )
    # The previous site's gates and the carried collapse, from the composed
    # step on an earlier stream.
    prev_fn, prev_base, prev_scale = site_params(rng, _HIDDEN)
    prev_stream = round_bf16(
        (rng.standard_normal((_ROWS, HC, _HIDDEN)) * 0.3).astype(np.float32)
    )
    post, comb, pre_mix, _ = mhc_pre(
        to_bf16_device(prev_stream),
        to_f32_device(prev_fn),
        to_f32_device(prev_scale),
        to_f32_device(prev_base),
        HC,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    fn, base, scale = site_params(rng, _HIDDEN)
    norm = RMSNorm(_HIDDEN, eps=NORM_EPS)
    norm.weight = to_bf16_device(
        round_bf16(1.0 + 0.1 * rng.standard_normal(_HIDDEN).astype(np.float32))
    )
    args = (
        to_f32_device(fn),
        to_f32_device(scale),
        to_f32_device(base),
    )

    got_stream, got_post, got_comb, got_pre, got_x, _ = mhc_fused_post_pre(
        to_bf16_device(stream),
        to_bf16_device(y),
        post,
        comb,
        pre_mix,
        *args,
        norm,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )

    want_stream = mhc_post(
        to_bf16_device(y), to_bf16_device(stream), post, comb, HC
    )
    want_post, want_comb, want_pre, collapsed = mhc_pre(
        want_stream,
        *args,
        HC,
        pre_mix,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    want_x = norm(collapsed)

    def host(t: Tensor) -> np.ndarray:
        return t.cast(DType.float32).to_numpy()

    for name, got, want, tol in (
        ("stream", got_stream, want_stream, 1e-2),
        ("post", got_post, want_post, 2e-3),
        ("comb", got_comb, want_comb, 2e-3),
        ("pre", got_pre, want_pre, 2e-3),
        ("normed", got_x, want_x, 1e-2),
    ):
        g, w = host(got), host(want)
        _report(name, g, w)
        np.testing.assert_allclose(g, w, rtol=tol, atol=tol)


def test_mxfp8_step_is_the_plain_step_with_the_quantize_folded_in() -> None:
    """The MXFP8 variant changes no result, and its rows feed a projection
    the operands the projection's own quantize would have produced."""
    rng = np.random.default_rng(51)
    stream = to_bf16_device(
        round_bf16(
            (rng.standard_normal((_ROWS, HC, _HIDDEN)) * 0.3).astype(np.float32)
        )
    )
    y = to_bf16_device(
        round_bf16(
            (rng.standard_normal((_ROWS, _HIDDEN)) * 0.5).astype(np.float32)
        )
    )
    prev_fn, prev_base, prev_scale = site_params(rng, _HIDDEN)
    post, comb, pre_mix, _ = mhc_pre(
        to_bf16_device(
            round_bf16(
                (rng.standard_normal((_ROWS, HC, _HIDDEN)) * 0.3).astype(
                    np.float32
                )
            )
        ),
        to_f32_device(prev_fn),
        to_f32_device(prev_scale),
        to_f32_device(prev_base),
        HC,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    fn, base, scale = site_params(rng, _HIDDEN)
    norm = RMSNorm(_HIDDEN, eps=NORM_EPS)
    norm.weight = to_bf16_device(
        round_bf16(1.0 + 0.1 * rng.standard_normal(_HIDDEN).astype(np.float32))
    )
    args = (
        stream,
        y,
        post,
        comb,
        pre_mix,
        to_f32_device(fn),
        to_f32_device(scale),
        to_f32_device(base),
        norm,
        NORM_EPS,
        HC_EPS,
        SINKHORN_ITERS,
    )

    *plain, plain_rows = mhc_fused_post_pre(*args)
    assert plain_rows is None
    *fused, rows = mhc_fused_post_pre(*args, mxfp8=True)
    assert rows is not None

    def host(t: Tensor) -> np.ndarray:
        return t.cast(DType.float32).to_numpy()

    for got, want in zip(fused, plain, strict=True):
        np.testing.assert_array_equal(host(got), host(want))

    out_dim = 256
    linear = MXFP8Linear(_HIDDEN, out_dim)
    linear.weight = (
        Tensor.from_dlpack(
            np.clip(rng.standard_normal((out_dim, _HIDDEN)), -4, 4).astype(
                np.float32
            )
        )
        .to(Accelerator())
        .cast(DType.float8_e4m3fn)
    )
    linear.scale = (
        Tensor.from_dlpack(
            np.power(
                2.0, rng.integers(-4, 1, (out_dim // 32, _HIDDEN // 32))
            ).astype(np.float32)
        )
        .to(Accelerator())
        .cast(DType.float8_e8m0fnu)
    )
    np.testing.assert_array_equal(
        host(linear.forward_quantized(rows)), host(linear(plain[4]))
    )


def _tap_of(x: np.ndarray) -> np.ndarray:
    total = x[:, 0].astype(np.float32)
    for i in range(1, x.shape[1]):
        total = total + x[:, i].astype(np.float32)
    rounded = torch.from_numpy(total * (1.0 / x.shape[1]))
    return rounded.to(torch.bfloat16).to(torch.float32).numpy()


def test_the_tap_reads_the_materialized_entry_stream() -> None:
    rng = np.random.default_rng(52)
    stream_np = round_bf16(np.full((_ROWS, HC, _HIDDEN), 0.25, np.float32))
    y_np = round_bf16(np.full((_ROWS, _HIDDEN), 1.5, np.float32))
    prev_fn, prev_base, prev_scale = site_params(rng, _HIDDEN)
    post, comb, pre_mix, _ = mhc_pre(
        to_bf16_device(
            round_bf16(
                (rng.standard_normal((_ROWS, HC, _HIDDEN)) * 0.3).astype(
                    np.float32
                )
            )
        ),
        to_f32_device(prev_fn),
        to_f32_device(prev_scale),
        to_f32_device(prev_base),
        HC,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    fn, base, scale = site_params(rng, _HIDDEN)
    norm = RMSNorm(_HIDDEN, eps=NORM_EPS)
    norm.weight = to_bf16_device(
        round_bf16(1.0 + 0.1 * rng.standard_normal(_HIDDEN).astype(np.float32))
    )
    stream = to_bf16_device(stream_np)
    y = to_bf16_device(y_np)
    args = (
        to_f32_device(fn),
        to_f32_device(scale),
        to_f32_device(base),
    )

    def host(t: Tensor) -> np.ndarray:
        return t.cast(DType.float32).to_numpy()

    entry, *_ = mhc_fused_post_pre(
        stream,
        y,
        post,
        comb,
        pre_mix,
        *args,
        norm,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    materialized = mhc_post(y, stream, post, comb, HC)

    tap = host(_hc_mean(entry))
    want = _tap_of(host(materialized))
    entry_host = host(entry)
    np.testing.assert_allclose(tap, want, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(
        entry_host, host(materialized), rtol=1e-2, atol=1e-2
    )

    ident_y, ident_post, ident_comb = mhc_identity_writeback(stream)
    entry0, *_ = mhc_fused_post_pre(
        stream,
        ident_y,
        ident_post,
        ident_comb,
        pre_mix,
        *args,
        norm,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=SINKHORN_ITERS,
    )
    np.testing.assert_allclose(
        host(_hc_mean(entry0)), _tap_of(host(stream)), rtol=1e-2, atol=1e-2
    )
