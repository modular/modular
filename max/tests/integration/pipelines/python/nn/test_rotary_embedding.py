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
from context_utils import create_text_context
from hypothesis import assume
from max.driver import CPU, Tensor
from max.dtype import DType
from max.graph import Dim, Graph, TensorType, TensorValue, TensorValueLike, ops
from max.nn import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    RotaryEmbedding,
)
from max.nn.kernels import fused_qk_ragged_rope
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from modular_graph_test import are_all_tensor_values, modular_graph_test

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
def test_freqs_cis(session, dtype: DType, params: RopeParams):
    with Graph("freqs_cis", input_types=[]) as graph:
        rope = RotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            MAX_SEQ_LEN,
        )
        graph.output(rope.freqs_cis)
        model = session.load(graph)
    result = model.execute()[0].to_numpy()
    # freq_cis result is stacked along a new dimension - real goes first, then imaginary.
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
    session,
    dtype: DType,
    base_params: RopeParams,
    scaling_params: Llama3RopeScalingParams,
):
    with Graph("freqs_cis", input_types=[]) as graph:
        rope = Llama3RotaryEmbedding(
            base_params.dim,
            base_params.n_heads,
            base_params.theta,
            MAX_SEQ_LEN,
            scaling_params=scaling_params,
        )
        graph.output(rope.freqs_cis)
        model = session.load(graph)
    result = model.execute()[0].to_numpy()
    d0, d1 = result.shape
    result = result.reshape(d0, d1 // 2, 2)
    # freq_cis result is stacked along a new dimension - real goes first, then imaginary.
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


class CannedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, freqs_cis: TensorValueLike):
        self._freqs_cis = freqs_cis


def torch_rope(x, freqs_cis, cache):
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
    [TensorType(DType.float32, ["batch", "seqlen", "n_kv_heads", 32])],
)
@pytest.mark.parametrize("start_pos", [0, 15])
def test_rope(session, input_type: TensorType, start_pos: Dim):
    _, seqlen, _, head_dim = input_type.shape
    freqs_cis_type = TensorType(input_type.dtype, [MAX_SEQ_LEN, head_dim])
    cachelike = TensorType(DType.int64, [start_pos])
    with Graph(
        "rope", input_types=[input_type, freqs_cis_type, cachelike]
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, freqs_cis, cache = graph.inputs
        freqs_cis = freqs_cis.reshape((MAX_SEQ_LEN, -1, 2))  # as complex
        start_pos_val = TensorValue(cache.shape[0])
        seq_len = x.shape[1]
        rope = CannedRotaryEmbedding(freqs_cis)
        graph.output(rope(x, start_pos_val, seq_len))

        @modular_graph_test(session, graph, max_magnitude=1.0)
        def test_correctness(execute, inputs, torch_inputs):
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


def test_kv_cache_ragged_rope(session):
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
        DType.float32, ["total_seq_len", num_q_heads, kv_params.head_dim]
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [
            "input_row_offsets_len",
        ],
    )

    freqs_cis_type = TensorType(
        input_type.dtype, [MAX_SEQ_LEN, kv_params.head_dim]
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
    seq_ids = []
    for _ in range(batch_size):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    input_row_offsets = Tensor(
        [batch_size + 1],
        DType.uint32,
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
    def test_runs_without_nan(execute, inputs, torch_inputs):
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
        assert np.any(result != np.nan)
        assert np.any(result != np.inf)
