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
"""CPU reference tests for the V4.1 tail RoPE.

The oracle rotates each interleaved pair as a complex number in fp32,
matching HF's ``view_as_complex`` pairing; the frequencies are checked
against the HF recipe computed independently, including the YaRN branch.
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
    V41RotaryEmbedding,
    apply_rope_tail,
)


def _to_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


_PARAMS = RopeParams(
    rope_head_dim=64,
    yarn_beta_fast=32.0,
    yarn_beta_slow=1.0,
    yarn_original_max_position=4096,
)


def test_inv_freqs_plain_matches_hf_recipe() -> None:
    """No-YaRN frequencies: ``1 / theta ** (arange(0, 64, 2) / 64)`` fp32."""
    rotary = V41RotaryEmbedding(_PARAMS, theta=10000.0, yarn_factor=1.0)
    f32 = np.float32
    expected = f32(1.0) / np.power(
        f32(10000.0), np.arange(0, 64, 2, dtype=f32) / f32(64)
    )
    np.testing.assert_array_equal(rotary.inv_freqs(), expected)


def test_inv_freqs_yarn_branch() -> None:
    """YaRN: the smooth band interpolates toward ``freqs / factor``."""
    rotary = V41RotaryEmbedding(_PARAMS, theta=10000.0, yarn_factor=16.0)
    got = rotary.inv_freqs()
    f32 = np.float32
    base = f32(1.0) / np.power(
        f32(10000.0), np.arange(0, 64, 2, dtype=f32) / f32(64)
    )
    dim = 64
    original = 4096

    def corrected_dim(rotations: float) -> float:
        return (
            dim
            * math.log(original / (rotations * 2 * math.pi))
            / (2 * math.log(10000.0))
        )

    low = max(math.floor(corrected_dim(32.0)), 0)
    high = min(math.ceil(corrected_dim(1.0)), dim - 1)
    ramp = np.clip(
        (np.arange(dim // 2, dtype=f32) - f32(low))
        / f32(max(high - low, 1e-3)),
        f32(0),
        f32(1),
    )
    smooth = f32(1) - ramp
    expected = base / f32(16.0) * (f32(1) - smooth) + base * smooth
    np.testing.assert_array_equal(got, expected)
    # Unit magnitude: no mscale anywhere.
    table = rotary.cos_sin(Tensor.from_dlpack(np.arange(8, dtype=np.int64)))
    cos_sin = F.cast(table, DType.float32).to_numpy()
    np.testing.assert_allclose(
        cos_sin[..., 0] ** 2 + cos_sin[..., 1] ** 2, 1.0, rtol=1e-5, atol=1e-6
    )


def _rotation_oracle(
    x_bf: np.ndarray, cos_sin: np.ndarray, rope_head_dim: int, inverse: bool
) -> np.ndarray:
    """Rotates the trailing pairs as complex numbers, in fp32, then bf16."""
    rows, width = x_bf.shape
    head, tail = (
        x_bf[:, : width - rope_head_dim],
        x_bf[:, width - rope_head_dim :],
    )
    pairs = tail.reshape(rows, rope_head_dim // 2, 2).astype(np.float32)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos = cos_sin[..., 0]  # [rows, pairs]
    sin = cos_sin[..., 1]
    if inverse:
        sin = -sin
    out_real = _to_bf16(real * cos - imag * sin)
    out_imag = _to_bf16(real * sin + imag * cos)
    out = np.stack([out_real, out_imag], axis=-1).reshape(rows, rope_head_dim)
    return np.concatenate([head, out], axis=1)


def test_apply_rope_tail_production_shape() -> None:
    """512-wide bf16 rows, 64-dim tail, interleaved pairing, no mscale."""
    rng = np.random.default_rng(6)
    rotary = V41RotaryEmbedding(_PARAMS, theta=10000.0, yarn_factor=16.0)
    positions = np.array([0, 1, 7, 1000, 4095], dtype=np.int64)
    x = (rng.standard_normal((5, 512)) * 0.5).astype(np.float32)
    x_bf = _to_bf16(x)

    cos_sin = rotary.cos_sin(Tensor.from_dlpack(positions))
    x_t = F.cast(Tensor.from_dlpack(x_bf), DType.bfloat16)
    out = apply_rope_tail(x_t, cos_sin, 64)
    assert out.dtype == DType.bfloat16
    got = F.cast(out, DType.float32).to_numpy()

    oracle_cos_sin = rotary.inv_freqs() * positions[:, None].astype(np.float32)
    expected = _rotation_oracle(
        x_bf,
        np.stack(
            [np.cos(oracle_cos_sin), np.sin(oracle_cos_sin)], axis=-1
        ).astype(np.float32),
        64,
        inverse=False,
    )
    np.testing.assert_array_equal(got, expected)


def test_apply_rope_tail_inverse_is_conjugate() -> None:
    """The inverse tail rotates by the conjugate: the oracle pins the sign."""
    rng = np.random.default_rng(7)
    rotary = V41RotaryEmbedding(_PARAMS, theta=10000.0, yarn_factor=1.0)
    positions = np.array([3, 511], dtype=np.int64)
    x = (rng.standard_normal((2, 512)) * 0.5).astype(np.float32)
    x_bf = _to_bf16(x)

    cos_sin = rotary.cos_sin(Tensor.from_dlpack(positions))
    x_t = F.cast(Tensor.from_dlpack(x_bf), DType.bfloat16)
    out = F.cast(
        apply_rope_tail(x_t, cos_sin, 64, inverse=True), DType.float32
    ).to_numpy()

    oracle_cos_sin = rotary.inv_freqs() * positions[:, None].astype(np.float32)
    expected = _rotation_oracle(
        x_bf,
        np.stack(
            [np.cos(oracle_cos_sin), np.sin(oracle_cos_sin)], axis=-1
        ).astype(np.float32),
        64,
        inverse=True,
    )
    np.testing.assert_array_equal(out, expected)
    # Head dims pass through untouched.
    np.testing.assert_array_equal(out[:, :448], x_bf[:, :448])


def test_head_passthrough_and_pair_norm() -> None:
    """Rotation preserves each rotated pair's norm (fp32 math, bf16 grid)."""
    rng = np.random.default_rng(8)
    rotary = V41RotaryEmbedding(_PARAMS, theta=10000.0, yarn_factor=1.0)
    positions = np.array([13, 777], dtype=np.int64)
    x = (rng.standard_normal((2, 512)) * 2.0).astype(np.float32)
    x_bf = _to_bf16(x)

    cos_sin = rotary.cos_sin(Tensor.from_dlpack(positions))
    out = F.cast(
        apply_rope_tail(
            F.cast(Tensor.from_dlpack(x_bf), DType.bfloat16), cos_sin, 64
        ),
        DType.float32,
    ).to_numpy()
    before = x_bf[:, 448:].reshape(2, 32, 2).astype(np.float32)
    after = out[:, 448:].reshape(2, 32, 2)
    norm_before = np.sum(before * before, axis=-1)
    norm_after = np.sum(after * after, axis=-1)
    np.testing.assert_allclose(norm_after, norm_before, rtol=2e-2, atol=1e-2)
    np.testing.assert_array_equal(out[:, :448], x_bf[:, :448])
