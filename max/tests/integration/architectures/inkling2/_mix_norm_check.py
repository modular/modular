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
"""The reference and runner shared by the inkling2_mix_norm GPU tests."""

from __future__ import annotations

import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.inkling2.layers.attn_res import (
    ATTN_RES_EPS,
    attn_res_mix_norm,
)
from max.tensor import Tensor, default_device

# Both sides round once to bfloat16 at the end. They reduce in different
# orders (per-thread partials against whole-axis sums), so one rounding can
# land a bfloat16 ulp apart: 2**-8 relative. Allow two ulps.
_RTOL = 2 * 2**-8
_ATOL = 2 * 2**-8
_NORM_EPS = 1e-6


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds float32 to the nearest bfloat16, ties to even, kept as float32."""
    bits = x.astype(np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def _reference(
    sources: list[np.ndarray], query: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    """``Inkling2RMSNorm(attn_res_mix(sources, query))`` in float32, numpy.

    The same math as the unfused ops: scores from the RMS-normalized sources,
    the normalized values mixed, the mixture kept in float32, and gamma
    applied before the single rounding to bfloat16.
    """
    hidden = sources[0].shape[-1]
    stack = np.stack(sources, axis=1)  # [tokens, sources, hidden]
    inv = 1.0 / np.sqrt((stack * stack).mean(-1, keepdims=True) + ATTN_RES_EPS)
    normed = stack * inv
    scores = (normed * query).sum(-1) * hidden**-0.5  # [tokens, sources]
    scores = scores - scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(-1, keepdims=True)
    mixed = (probs[..., None] * normed).sum(1)  # [tokens, hidden]
    inv_out = 1.0 / np.sqrt((mixed * mixed).mean(-1, keepdims=True) + _NORM_EPS)
    return _bf16(mixed * inv_out * gamma)


def run_case(
    hidden: int,
    tokens: int,
    num_blocks: int,
    partial_dtype: DType,
    num_queries: int,
    rows: tuple[int, ...] | None = None,
) -> None:
    rng = np.random.default_rng(hidden * 1000 + tokens * 10 + num_blocks)
    blocks_np = [
        _bf16(rng.standard_normal((tokens, hidden)).astype(np.float32))
        for _ in range(num_blocks)
    ]
    partial_np = rng.standard_normal((tokens, hidden)).astype(np.float32)
    if partial_dtype == DType.bfloat16:
        partial_np = _bf16(partial_np)
    queries_np = rng.standard_normal((num_queries, hidden)).astype(np.float32)
    gamma_np = _bf16(
        (1.0 + 0.1 * rng.standard_normal(hidden)).astype(np.float32)
    )

    with default_device(Accelerator(0)):
        blocks = [
            Tensor(b, dtype=DType.float32).cast(DType.bfloat16)
            for b in blocks_np
        ]
        partial = Tensor(partial_np, dtype=DType.float32).cast(partial_dtype)
        queries = Tensor(queries_np, dtype=DType.float32)
        gamma = Tensor(gamma_np, dtype=DType.float32).cast(DType.bfloat16)
        for row in range(num_queries) if rows is None else rows:
            want = _reference(
                [*blocks_np, partial_np], queries_np[row], gamma_np
            )
            got = (
                attn_res_mix_norm(
                    blocks,
                    partial,
                    queries,
                    gamma,
                    query_row=row,
                    norm_eps=_NORM_EPS,
                    normalize_values=True,
                )
                .cast(DType.float32)
                .to_numpy()
            )
            np.testing.assert_allclose(
                got,
                want,
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=(
                    f"hidden={hidden} tokens={tokens} blocks={num_blocks}"
                    f" partial={partial_dtype} row={row}"
                ),
            )
