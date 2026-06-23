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
"""Correctness tests for the FP8 MLA load-time scale gather.

GLM-5.1-FP8 has `qk_nope_head_dim = 192` and `v_head_dim = 256`;
per-head row count `Dn + Dv = 448` is not a multiple of the 128-row
on-disk scale block extent, so 128-row scale blocks straddle the K/V
split inside a head and the boundary between consecutive heads. The
pre-existing per-head reshape `[R/128, H, -1]` and the K/V slice at
`Dn // 128` are unsatisfiable.

These tests pin the gather formula mirrored in
:class:`LatentAttentionWithRopeFp8._b_scale_granularity` and
`_gather_per_head_scale` against direct on-disk dequant, for both
DeepSeek (granularity=128, no replication — regression) and GLM
(granularity=64, replicated scalars) geometries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _scale_granularity(dn: int, dv: int, block_m: int = 128) -> int:
    """NumPy mirror of `LatentAttentionWithRopeFp8._b_scale_granularity`."""
    residue = (dn + dv) % block_m
    if residue == 0 and dn % block_m == 0:
        return block_m
    return math.gcd(residue, block_m)


def _per_head_scale(
    flat_scale: np.ndarray,
    *,
    start_row: int,
    n_rows: int,
    granularity: int,
    block_m: int,
) -> np.ndarray:
    """NumPy mirror of `LatentAttentionWithRopeFp8._gather_per_head_scale`."""
    n_chunks = (n_rows + granularity - 1) // granularity
    rows = [(start_row + k * granularity) // block_m for k in range(n_chunks)]
    return flat_scale[rows]


def _on_disk_dequant(
    weight: np.ndarray, flat_scale: np.ndarray, block_m: int, block_k: int
) -> np.ndarray:
    """Reference dequant: `w_real[i,j] = w_fp8[i,j] * scale[i//Bm, j//Bk]`."""
    M, K = weight.shape
    scale_full = np.repeat(
        np.repeat(flat_scale, block_m, axis=0), block_k, axis=1
    )
    return weight * scale_full[:M, :K]


def test_granularity_choice() -> None:
    """DeepSeek dims yield granularity 128 (unchanged); GLM dims yield 64."""
    assert _scale_granularity(128, 128) == 128  # DeepSeek
    assert _scale_granularity(192, 256) == 64  # GLM-5.1-FP8


@pytest.mark.parametrize(
    "h,dn,dv", [(4, 128, 128), (4, 192, 256), (128, 192, 256)]
)
def test_gather_dequant_matches_on_disk(h: int, dn: int, dv: int) -> None:
    """Dequantizing `kv_b_proj` through the per-head gathered scales
    produces the same matrix as direct elementwise dequant via the flat
    on-disk scale, for both DeepSeek (H=4) and GLM (H=4 and the
    production H=128) geometries.
    """
    block_m = block_k = 128
    r = 512
    per_head = dn + dv
    g = _scale_granularity(dn, dv)
    rng = np.random.default_rng(seed=2758)
    weight = rng.integers(-127, 127, size=(h * per_head, r)).astype(np.float32)
    flat_scale = rng.uniform(
        0.25, 4.0, size=(h * per_head // block_m, r // block_k)
    ).astype(np.float32)

    ref = _on_disk_dequant(weight, flat_scale, block_m, block_k)
    got = np.zeros_like(ref)
    for head in range(h):
        for start, n in (
            (head * per_head, dn),
            (head * per_head + dn, dv),
        ):
            scale = _per_head_scale(
                flat_scale,
                start_row=start,
                n_rows=n,
                granularity=g,
                block_m=block_m,
            )
            for kn in range((n + g - 1) // g):
                d_lo, d_hi = kn * g, min(kn * g + g, n)
                for kk in range(r // block_k):
                    col_lo, col_hi = kk * block_k, (kk + 1) * block_k
                    got[start + d_lo : start + d_hi, col_lo:col_hi] = (
                        weight[start + d_lo : start + d_hi, col_lo:col_hi]
                        * scale[kn, kk]
                    )

    np.testing.assert_array_equal(got, ref)
