# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import assume
from max.driver import CPU, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValueLike, ops
from max.nn import (
    DynamicRotaryEmbedding,
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    LongRoPEScalingParams,
    RotaryEmbedding,
)
from max.nn.kernels import fused_qk_ragged_rope
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from modular_graph_test import are_all_tensor_values, modular_graph_test
from test_common.context_utils import create_text_context

MAX_SEQ_LEN = 2**16
ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-7


def torch_freqs_cis(dim: int, theta: float):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(
        MAX_SEQ_LEN * 2.0, device=freqs.device, dtype=torch.float32
    )
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def torch_llama3_freqs_cis(
    dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    orig_max_position: int,
):
    inv_freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    low_freq_wavelen = orig_max_position / low_freq_factor
    high_freq_wavelen = orig_max_position / high_freq_factor

    wave_len = 2 * math.pi / inv_freqs
    if low_freq_factor != high_freq_factor:
        smooth = (orig_max_position / wave_len - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = 0
    freqs = torch.where(
        wave_len < high_freq_wavelen,
        inv_freqs,
        torch.where(
            wave_len > low_freq_wavelen,
            inv_freqs / factor,
            (1 - smooth) * inv_freqs / factor + smooth * inv_freqs,
        ),
    )
    t = torch.arange(
        MAX_SEQ_LEN * 2.0, device=freqs.device, dtype=torch.float32
    )
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def torch_dynamic_rope_freqs_cis(dim: int, theta: float, max_seq_len: int):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    t = torch.arange(max_seq_len * 2.0, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


@dataclass
class RopeParams:
    dim: int
    n_heads: int
    theta: float

    @property
    def head_dim(self):
        return self.dim // self.n_heads


@pytest.mark.parametrize(
    "params",
    [
        RopeParams(dim=64, n_heads=4, theta=1e4),
        RopeParams(dim=512, n_heads=16, theta=5e5),
    ],
)
@pytest.mark.parametrize("dtype", [DType.float32])
def test_freqs_cis(session, dtype: DType, params: RopeParams) -> None:  # noqa: ANN001
    with Graph("freqs_cis", input_types=[]) as graph:
        rope = RotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            MAX_SEQ_LEN,
            head_dim=params.head_dim,
            device=DeviceRef.CPU(),
        )
        graph.output(rope.freqs_cis)
        model = session.load(graph)
    result = model.execute()[0].to_numpy()

    # Handle flattened freqs_cis format - reshape back to 3D to extract real/imaginary
    if len(result.shape) == 2:
        d0, d1 = result.shape  # (max_seq_len * 2, head_dim)
        result = result.reshape(
            (d0, d1 // 2, 2)
        )  # (max_seq_len * 2, head_dim // 2, 2)

    # freqs_cis result is stacked along a new dimension - real goes first, then imaginary.
    # The result is a tensor with shape (..., 2) where the last dimension holds [real, imaginary]
    # We extract and convert into a complex tensor type before comparing them.
    result_cis_complex = result[:, :, 0] + 1j * result[:, :, 1]
    expected = torch_freqs_cis(params.head_dim, params.theta)
    np.testing.assert_allclose(
        result_cis_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "base_params",
    [
        RopeParams(
            dim=64,
            n_heads=4,
            theta=1e4,
        ),
        RopeParams(
            dim=512,
            n_heads=16,
            theta=5e5,
        ),
    ],
)
@pytest.mark.parametrize(
    "scaling_params",
    [
        Llama3RopeScalingParams(
            factor=4.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=8192,
        ),
        Llama3RopeScalingParams(
            factor=8.0,
            low_freq_factor=4.0,
            high_freq_factor=4.0,
            orig_max_position=8192,
        ),
    ],
)
@pytest.mark.parametrize("dtype", [DType.float32])
def test_llama3_freqs_cis(
    session,  # noqa: ANN001
    dtype: DType,
    base_params: RopeParams,
    scaling_params: Llama3RopeScalingParams,
) -> None:
    with Graph("freqs_cis", input_types=[]) as graph:
        rope = Llama3RotaryEmbedding(
            base_params.dim,
            base_params.n_heads,
            base_params.theta,
            MAX_SEQ_LEN,
            scaling_params=scaling_params,
            head_dim=base_params.head_dim,
            device=DeviceRef.CPU(),
        )
        graph.output(rope.freqs_cis)
        model = session.load(graph)
    result = model.execute()[0].to_numpy()
    d0, d1 = result.shape
    result = result.reshape(d0, d1 // 2, 2)
    # freqs_cis result is stacked along a new dimension - real goes first, then imaginary.
    # The result is a tensor with shape (..., 2) where the last dimension holds [real, imaginary]
    # We extract and convert into a complex tensor type before comparing them.
    result_cis_complex = result[:, :, 0] + 1j * result[:, :, 1]
    expected = torch_llama3_freqs_cis(
        base_params.head_dim,
        base_params.theta,
        scaling_params.factor,
        scaling_params.low_freq_factor,
        scaling_params.high_freq_factor,
        scaling_params.orig_max_position,
    )
    np.testing.assert_allclose(
        result_cis_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "dim, n_heads, theta, short_seq_len, long_seq_len",
    [
        (64, 4, 1e4, 4096, 8192),
        (512, 16, 5e5, 8192, 16384),
    ],
)
def test_dynamic_rope_freqs_cis(
    session,  # noqa: ANN001
    dim: int,
    n_heads: int,
    theta: float,
    short_seq_len: int,
    long_seq_len: int,
) -> None:
    """Test that DynamicRotaryEmbedding behaves identically to RotaryEmbedding
    for short sequences, and correctly expands the freqs_cis buffer for long
    sequences."""
    head_dim = dim // n_heads

    # Test short sequence: should have the same behavior as default RoPE.
    with Graph("dynamic_rope_short", input_types=[]) as graph:
        rope = DynamicRotaryEmbedding(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=short_seq_len,
            head_dim=head_dim,
            device=DeviceRef.CPU(),
        )
        graph.output(rope.freqs_cis)
        model = session.load(graph)

    # Manually reshape and recombine the real and imaginary components into a
    # complex-valued array for comparison against the expected result.
    result = model.execute()[0].to_numpy()
    d0, d1 = result.shape
    result = result.reshape((d0, d1 // 2, 2))
    result_complex = result[:, :, 0] + 1j * result[:, :, 1]

    expected = torch_dynamic_rope_freqs_cis(head_dim, theta, short_seq_len)
    np.testing.assert_allclose(
        result_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )

    # Test long sequence: should dynamically expand.
    with Graph("dynamic_rope_long", input_types=[]) as graph:
        rope = DynamicRotaryEmbedding(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=short_seq_len,
            head_dim=head_dim,
            device=DeviceRef.CPU(),
        )
        # Simulate runtime position_ids that require growing buffer.
        dummy_position_ids = ops.range(0, long_seq_len, 1, dtype=DType.int64)
        rope.maybe_update_freqs(dummy_position_ids)
        graph.output(rope.freqs_cis)
        model = session.load(graph)

    # Manually reshape and recombine the real and imaginary components into a
    # complex-valued array for comparison against the expected result.
    result = model.execute()[0].to_numpy()
    d0, d1 = result.shape
    result = result.reshape((d0, d1 // 2, 2))
    result_complex = result[:, :, 0] + 1j * result[:, :, 1]

    expected = torch_dynamic_rope_freqs_cis(head_dim, theta, long_seq_len)
    np.testing.assert_allclose(
        result_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


class CannedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, freqs_cis: TensorValueLike) -> None:
        self._freqs_cis = freqs_cis


def torch_rope(x, freqs_cis, cache):  # noqa: ANN001
    start_pos = cache.shape[0]
    seq_len = x.shape[1]
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    freqs_cis = torch.view_as_complex(freqs_cis.reshape(seq_len, -1, 2))
    return apply_rotary_emb(x, freqs_cis)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, x_)
    return torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(
            DType.float32,
            ["batch", "seqlen", "n_kv_heads", 32],
            device=DeviceRef.CPU(),
        )
    ],
)
@pytest.mark.parametrize("start_pos", [0, 15])
def test_rope(session, input_type: TensorType, start_pos: Dim) -> None:  # noqa: ANN001
    _, seqlen, _, head_dim = input_type.shape
    freqs_cis_type = TensorType(
        input_type.dtype, [MAX_SEQ_LEN, head_dim], device=DeviceRef.CPU()
    )
    cachelike = TensorType(DType.int64, [start_pos], device=DeviceRef.CPU())
    with Graph(
        "rope", input_types=[input_type, freqs_cis_type, cachelike]
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, freqs_cis, cache = graph.inputs
        freqs_cis = freqs_cis.reshape((MAX_SEQ_LEN, -1, 2))  # as complex
        start_pos = cache.shape[0]
        seq_len = x.shape[1]
        rope = CannedRotaryEmbedding(freqs_cis)
        graph.output(rope(x, start_pos, seq_len))

        @modular_graph_test(session, graph, max_magnitude=1.0)
        def test_correctness(execute, inputs, torch_inputs) -> None:  # noqa: ANN001
            x, freqs_cis, cache = inputs
            start_pos = cache.shape[0]
            seq_len = x.shape[1]
            assume(start_pos + seq_len < MAX_SEQ_LEN)
            result = execute(inputs).to_numpy()
            expected = torch_rope(*torch_inputs).detach().numpy()

            np.testing.assert_allclose(
                result,
                expected,
                atol=ACCURACY_ATOL,
                rtol=ACCURACY_RTOL,
                equal_nan=True,
            )


def test_kv_cache_ragged_rope(session) -> None:  # noqa: ANN001
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.float32,
        ["total_seq_len", num_q_heads, kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [
            "input_row_offsets_len",
        ],
        device=DeviceRef.CPU(),
    )

    freqs_cis_type = TensorType(
        input_type.dtype,
        [MAX_SEQ_LEN, kv_params.head_dim],
        device=DeviceRef.CPU(),
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_manager.input_symbols()[0]
    )

    def construct() -> Graph:
        with Graph(
            "call_ragged_qk_rope",
            input_types=[
                input_type,
                input_row_offsets_type,
                freqs_cis_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                is_cache_empty_type,
            ],
        ) as g:
            assert are_all_tensor_values(g.inputs)
            (
                input,
                input_row_offsets,
                freqs_cis,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs
            layer_idx = ops.constant(
                0,
                DType.uint32,
                DeviceRef.CPU(),
            )

            kv_collection = fetch_op(
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            )
            result = fused_qk_ragged_rope(
                kv_params,
                input,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                layer_idx,
            )
            g.output(result)
        return g

    g = construct()

    # Claim seq_ids in cache
    seq_ids_to_claim = list(kv_manager.available)[:batch_size]
    kv_manager.external_claim(seq_ids_to_claim)
    seq_ids = seq_ids_to_claim

    input_row_offsets = Tensor(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    batch = [
        create_text_context(s, np.empty(prompt_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(batch)[0]
    )

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offsets_len": len(prompt_lens) + 1,
        },
        provided_inputs={
            1: input_row_offsets,
            3: blocks,
            4: cache_lengths,
            5: lookup_table_tensor,
            6: is_cache_empty_buf,
        },
    )
    def test_runs_without_nan(execute, inputs, torch_inputs) -> None:  # noqa: ANN001
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
        assert np.any(result != np.nan)
        assert np.any(result != np.inf)


def torch_longrope_freqs_cis(
    dim: int,
    theta: float,
    max_seq_len: int,
    short_factor: list[float],
    long_factor: list[float],
    original_max_position: int,
):
    """PyTorch reference implementation of LongRoPE frequency computation with stitched table."""
    # Compute base inverse frequencies
    inv_freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )

    # Apply short scaling factors
    short_factors_tensor = torch.tensor(
        short_factor[: len(inv_freqs)], dtype=torch.float32
    )
    scaled_inv_freqs_short = inv_freqs / short_factors_tensor

    # Apply long scaling factors
    long_factors_tensor = torch.tensor(
        long_factor[: len(inv_freqs)], dtype=torch.float32
    )
    scaled_inv_freqs_long = inv_freqs / long_factors_tensor

    # Generate position ids for the "short" part (0 to original_max_position)
    t_short = torch.arange(original_max_position, dtype=torch.float32)

    # Generate position ids for the "long" part (original_max_position to max_seq_len*2)
    t_long = torch.arange(
        original_max_position, max_seq_len * 2.0, dtype=torch.float32
    )

    # Compute frequencies for both parts
    freqs_short = torch.outer(t_short, scaled_inv_freqs_short)
    freqs_long = torch.outer(t_long, scaled_inv_freqs_long)

    # Concatenate the two parts
    freqs_combined = torch.cat([freqs_short, freqs_long], dim=0)

    # Convert to complex
    freqs_cis = torch.polar(torch.ones_like(freqs_combined), freqs_combined)
    return freqs_cis


@pytest.mark.parametrize(
    "params",
    [
        RopeParams(dim=3072, n_heads=32, theta=10000.0),
        RopeParams(dim=2048, n_heads=16, theta=10000.0),
    ],
)
@pytest.mark.parametrize("dtype", [DType.float32])
def test_longrope_scaling(session, dtype: DType, params: RopeParams) -> None:  # noqa: ANN001
    """Test LongRoPE frequency scaling with different scaling factors for short and long sequences.

    This test verifies that LongRoPE correctly applies frequency scaling parameters:
    - short_factor: scaling factors for shorter sequences (default 1.0)
    - long_factor: scaling factors for longer sequences (2x the short factors)
    - Ensures proper shape and numerical stability of the frequency embeddings
    - Validates numerical correctness against PyTorch reference implementation
    """
    max_seq_len = 131072

    # Create scaling params with long factors being 2x short factors
    scaling_params = LongRoPEScalingParams(
        short_factor=[1.0] * (params.head_dim // 2),
        long_factor=[2.0] * (params.head_dim // 2),
        original_max_position=4096,
        max_position_embeddings=max_seq_len,
    )

    with Graph("longrope_freqs_cis", input_types=[]) as graph:
        rope = LongRoPERotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            max_seq_len,
            head_dim=params.head_dim,
            device=DeviceRef.CPU(),
            scaling_params=scaling_params,
        )
        graph.output(rope.freqs_cis)
        model = session.load(graph)

    result = model.execute()[0].to_numpy()

    # Basic shape and validity checks - enforce flattened 2D shape
    assert len(result.shape) == 2, (
        f"Expected 2D tensor, but got shape {result.shape}"
    )
    assert result.shape[0] == max_seq_len * 2
    assert result.shape[1] == params.head_dim
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Numerical validation against PyTorch reference
    # Reshape the validated 2D tensor to extract real/imaginary parts
    d0, d1 = result.shape  # (max_seq_len * 2, head_dim)
    result = result.reshape(
        (d0, d1 // 2, 2)
    )  # (max_seq_len * 2, head_dim // 2, 2)

    result_cis_complex = result[:, :, 0] + 1j * result[:, :, 1]
    expected = torch_longrope_freqs_cis(
        params.head_dim,
        params.theta,
        max_seq_len,
        scaling_params.short_factor,
        scaling_params.long_factor,
        scaling_params.original_max_position,
    )

    np.testing.assert_allclose(
        result_cis_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )

    # Test short sequence behavior (should use short_factor)
    short_max_seq_len = 2048  # Less than original_max_position=4096

    with Graph("longrope_short_seq", input_types=[]) as graph:
        rope_short = LongRoPERotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            short_max_seq_len,
            head_dim=params.head_dim,
            device=DeviceRef.CPU(),
            scaling_params=scaling_params,
        )
        graph.output(rope_short.freqs_cis)
        model_short = session.load(graph)

    result_short = model_short.execute()[0].to_numpy()

    # Validate short sequence uses short_factor
    if len(result_short.shape) == 2:
        d0, d1 = result_short.shape
        result_short = result_short.reshape((d0, d1 // 2, 2))

    result_short_complex = result_short[:, :, 0] + 1j * result_short[:, :, 1]
    expected_short = torch_longrope_freqs_cis(
        params.head_dim,
        params.theta,
        short_max_seq_len,
        scaling_params.short_factor,
        scaling_params.long_factor,
        scaling_params.original_max_position,
    )

    np.testing.assert_allclose(
        result_short_complex,
        expected_short,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )

    # Test without scaling (should behave like standard RoPE)
    with Graph("longrope_no_scaling", input_types=[]) as graph:
        rope_no_scale = LongRoPERotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            4096,  # smaller max_seq_len
            head_dim=params.head_dim,
            device=DeviceRef.CPU(),
            scaling_params=None,  # No scaling
        )
        graph.output(rope_no_scale.freqs_cis)
        model_no_scale = session.load(graph)

    result_no_scale = model_no_scale.execute()[0].to_numpy()

    # Should behave like standard RoPE when no scaling params
    assert result_no_scale.shape[0] == 4096 * 2
    assert result_no_scale.shape[1] == params.head_dim

    # Compare with standard RoPE for validation
    with Graph("standard_rope", input_types=[]) as graph:
        rope_standard = RotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            4096,
            head_dim=params.head_dim,
            device=DeviceRef.CPU(),
        )
        graph.output(rope_standard.freqs_cis)
        model_standard = session.load(graph)

    result_standard = model_standard.execute()[0].to_numpy()

    # LongRoPE without scaling should match standard RoPE
    np.testing.assert_allclose(
        result_no_scale,
        result_standard,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )
