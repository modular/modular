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
"""CPU reference tests for the mHC stream mixing and collapse steps.

The gate projection (``mo.hyper_connection.gates``) is GPU-only, so these
tests pin the pure-graph pieces every boundary is composed of: the
``post``/``comb`` write-back orientation, the carried-``pre`` collapse,
the final collapse and the identity write-back. The oracles are
independent NumPy transcriptions of the HF reference's mHC chain step
and the vLLM ``mhc_post_torch`` einsum.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import (
    mhc_final_collapse,
    mhc_identity_writeback,
    mhc_post,
)


def _to_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


_HC = 4
_HIDDEN = 128


def _mhc_post_oracle(
    y_bf: np.ndarray,
    stream_bf: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
) -> np.ndarray:
    """``post[j] * y + sum_i comb[i, j] * stream[i]``, fp32, one bf16 round."""
    y = y_bf.astype(np.float32)
    stream = stream_bf.astype(np.float32)
    mixed = np.einsum("rij,rih->rjh", comb.astype(np.float32), stream)
    out = post.astype(np.float32)[:, :, None] * y[:, None, :] + mixed
    return _to_bf16(out)


def _bf16(x: np.ndarray) -> Tensor:
    return F.cast(Tensor.from_dlpack(x), DType.bfloat16)


def test_mhc_post_comb_orientation() -> None:
    """comb rows index *input* streams, columns *output* streams."""
    rng = np.random.default_rng(9)
    rows = 5
    y = (rng.standard_normal((rows, _HIDDEN)) * 0.3).astype(np.float32)
    stream = (rng.standard_normal((rows, _HC, _HIDDEN)) * 0.3).astype(
        np.float32
    )
    post = rng.standard_normal((rows, _HC)).astype(np.float32)
    comb = rng.standard_normal((rows, _HC, _HC)).astype(np.float32)
    y_bf, stream_bf = _to_bf16(y), _to_bf16(stream)

    out = mhc_post(
        _bf16(y_bf),
        _bf16(stream_bf),
        Tensor.from_dlpack(post),
        Tensor.from_dlpack(comb),
        _HC,
    )
    got = F.cast(out, DType.float32).to_numpy()
    np.testing.assert_array_equal(
        got, _mhc_post_oracle(y_bf, stream_bf, post, comb)
    )
    # The transposed comb gives a different answer, so the orientation is
    # pinned, not vacuous.
    flipped = _mhc_post_oracle(
        y_bf, stream_bf, post, np.transpose(comb, (0, 2, 1))
    )
    assert not np.array_equal(got, flipped)


def test_mhc_final_collapse_uses_carried_pre() -> None:
    """The final collapse is the plain weighted sum over the stream axis."""
    rng = np.random.default_rng(10)
    rows = 7
    stream = (rng.standard_normal((rows, _HC, _HIDDEN)) * 0.4).astype(
        np.float32
    )
    pre = rng.standard_normal((rows, _HC)).astype(np.float32)
    stream_bf = _to_bf16(stream)

    out = mhc_final_collapse(_bf16(stream_bf), Tensor.from_dlpack(pre))
    got = F.cast(out, DType.float32).to_numpy()
    expected = _to_bf16(
        np.sum(
            pre.astype(np.float32)[:, :, None] * stream_bf.astype(np.float32),
            axis=1,
        )
    )
    np.testing.assert_array_equal(got, expected)


def test_identity_writeback_is_exact() -> None:
    """``y = 0, post = 0, comb = I`` leaves the stream bit-identical."""
    rng = np.random.default_rng(11)
    rows = 3
    stream = (rng.standard_normal((rows, _HC, _HIDDEN)) * 0.5).astype(
        np.float32
    )
    stream_bf = _to_bf16(stream)
    stream_t = _bf16(stream_bf)

    y, post, comb = mhc_identity_writeback(stream_t)
    assert y.dtype == DType.bfloat16
    assert post.dtype == DType.float32
    assert comb.dtype == DType.float32
    # comb is the row-major identity: [1, 0, ..., 1, ...].
    comb_np = comb.to_numpy()
    identity = np.eye(_HC, dtype=np.float32).reshape(-1)
    np.testing.assert_array_equal(comb_np[0], identity)

    out = mhc_post(y, stream_t, post, F.reshape(comb, [-1, _HC, _HC]), _HC)
    np.testing.assert_array_equal(
        F.cast(out, DType.float32).to_numpy(), stream_bf
    )
