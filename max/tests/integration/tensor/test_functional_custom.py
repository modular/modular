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
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn import kernels

DEVICE = Accelerator() if accelerator_count() else CPU()

moe_create_indices = F.functional(kernels.moe_create_indices)
scatter_set_constant = F.functional(kernels.scatter_set_constant)


def _reference_flash_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    mask_variant: kernels.MHAMaskVariant,
    scale: float,
) -> np.ndarray:
    q_heads = np.transpose(q.astype(np.float32), (0, 2, 1, 3))
    k_heads = np.transpose(k.astype(np.float32), (0, 2, 1, 3))
    v_heads = np.transpose(v.astype(np.float32), (0, 2, 1, 3))

    scores = np.matmul(q_heads, np.swapaxes(k_heads, -1, -2)) * scale

    if mask_variant == kernels.MHAMaskVariant.CAUSAL_MASK:
        causal_mask = np.triu(
            np.ones((q.shape[1], k.shape[1]), dtype=bool),
            k=1,
        )
        scores = np.where(causal_mask[None, None, :, :], -10000.0, scores)
    elif mask_variant != kernels.MHAMaskVariant.NULL_MASK:
        raise AssertionError(
            f"unsupported mask variant in test: {mask_variant}"
        )

    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    output = np.matmul(probs, v_heads)
    return np.transpose(output, (0, 2, 1, 3)).astype(q.dtype, copy=False)


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


@pytest.mark.skipif(
    DEVICE.is_host, reason="moe_create_indices only supports GPU devices"
)
def test_custom() -> None:
    indices = Tensor.ones([4], dtype=DType.int32, device=DEVICE)
    token_expert_order, *_rest = moe_create_indices(indices, 8)
    assert token_expert_order.real


@pytest.mark.skipif(
    DEVICE.is_host, reason="scatter_set_constant only supports GPU devices"
)
def test_inplace_custom() -> None:
    values = Tensor.zeros([2, 2])
    indices = Tensor.ones([1, 1], dtype=DType.int32)
    scatter_set_constant(values, indices, 5.0)
    assert values[1, 0].item() == 5.0
    assert values.real
    scatter_set_constant(values, indices, 4.0)
    assert not values.real
    assert values[1, 0].item() == 4.0
    assert values.real


def test_custom_with_custom_extensions(
    kernel_verification_ops_path: Path,
) -> None:
    """Test F.custom with inline custom_extensions loading."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    # Call custom op with custom_extensions - kernels loaded automatically
    result = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=kernel_verification_ops_path,
    )

    assert len(result) == 1
    output = result[0]
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.real


def test_custom_with_custom_extensions_list(
    kernel_verification_ops_path: Path,
) -> None:
    """Test F.custom with custom_extensions as a list."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    result = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=[kernel_verification_ops_path],
    )

    assert len(result) == 1
    assert result[0].real


def test_custom_with_string_path(kernel_verification_ops_path: Path) -> None:
    """Test F.custom with custom_extensions as a string path."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    result = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=str(kernel_verification_ops_path),
    )

    assert len(result) == 1
    assert result[0].real


def test_custom_extensions_cached_across_calls(
    kernel_verification_ops_path: Path,
) -> None:
    """Test that custom_extensions are cached and not reloaded on every call."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    # First call
    result1 = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=kernel_verification_ops_path,
    )
    assert result1[0].real

    # Second call - should use cached extension
    result2 = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=kernel_verification_ops_path,
    )
    assert result2[0].real


def test_custom_helper_function_pattern(
    kernel_verification_ops_path: Path,
) -> None:
    """Test the recommended pattern for creating reusable custom op wrappers."""

    def my_add(a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition using custom Mojo kernel."""
        return F.custom(
            "my_add",
            device=a.device,
            values=[a, b],
            out_types=[a.type],
            custom_extensions=kernel_verification_ops_path,
        )[0]

    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.full([64], 2.0, dtype=DType.float32, device=CPU())

    result = my_add(x, y)

    assert result.real
    assert result.shape == x.shape


@pytest.mark.parametrize(
    "mask_variant",
    [
        kernels.MHAMaskVariant.NULL_MASK,
        kernels.MHAMaskVariant.CAUSAL_MASK,
    ],
)
def test_flash_attention_gpu_cpu_fallback(
    mask_variant: kernels.MHAMaskVariant,
) -> None:
    batch, seq_len, num_heads, head_dim = 1, 8, 2, 16
    scale = 0.25

    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(
        np.float32
    )
    k_np = rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(
        np.float32
    )
    v_np = rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(
        np.float32
    )

    q = Tensor(q_np, device=CPU())
    k = Tensor(k_np, device=CPU())
    v = Tensor(v_np, device=CPU())

    @F.functional
    def run_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return kernels.flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=mask_variant,
            scale=scale,
        )

    output = run_attention(q, k, v)

    assert output.real
    assert output.shape == q.shape
    assert output.dtype == q.dtype
    np.testing.assert_allclose(
        np.from_dlpack(output),
        _reference_flash_attention(
            q_np,
            k_np,
            v_np,
            mask_variant=mask_variant,
            scale=scale,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_flash_attention_gpu_rejects_valid_length_on_cpu() -> None:
    q = Tensor.ones([1, 8, 2, 16], dtype=DType.float32, device=CPU())
    k = Tensor.ones([1, 8, 2, 16], dtype=DType.float32, device=CPU())
    v = Tensor.ones([1, 8, 2, 16], dtype=DType.float32, device=CPU())
    valid_length = Tensor([8], dtype=DType.uint32, device=CPU())

    @F.functional
    def run_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        valid_length: Tensor,
    ) -> Tensor:
        return kernels.flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=kernels.MHAMaskVariant.NULL_MASK,
            scale=0.25,
            valid_length=valid_length,
        )

    with pytest.raises(
        ValueError,
        match="padded CPU fallback is not implemented",
    ):
        run_attention(q, k, v, valid_length)
