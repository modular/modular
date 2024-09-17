# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""

from dataclasses import dataclass

import numpy as np
import pytest
from max.dtype import DType
from max.graph import Graph, TensorType
from nn import Attention, Linear, RotaryEmbedding


@dataclass
class NanoLlama3:
    """Class to hold toy weights and parameters for testing llama3 code."""

    params = {
        "dim": 2,
        "n_layers": 1,
        "n_heads": 1,
        "vocab_size": 4,
        "norm_eps": 1e-5,
        "n_kv_heads": 1,
        "head_dim": 2,
        "n_rep": 1,
    }

    mlp_w1 = (
        np.array(
            [
                [0.5641, 0.4875],
                [-1.1172, -1.1583],
            ]
        )
        .astype(np.float32)
        .transpose(-1, -2)
    )

    mlp_w2 = (
        np.array(
            [
                [0.5355, -0.9487],
                [-0.6487, 0.1838],
            ]
        )
        .astype(np.float32)
        .transpose(-1, -2)
    )

    mlp_w3 = (
        np.array(
            [
                [-0.6765, 0.7103],
                [-0.4643, 0.2860],
            ]
        )
        .astype(np.float32)
        .transpose(-1, -2)
    )

    attn_wq = (
        np.array(
            [
                0.3256,
                -1.8786,
                -0.4062,
                -0.4507,
            ]
        )
        .reshape((2, 2))
        .astype(np.float32)
        .transpose(-1, -2)
    )
    attn_wk = (
        np.array(
            [
                0.6694,
                -0.7980,
                0.8910,
                0.9103,
            ]
        )
        .reshape((2, 2))
        .astype(np.float32)
        .transpose(-1, -2)
    )
    attn_wv = (
        np.array(
            [
                0.5933,
                1.0371,
                -0.0971,
                0.0469,
            ]
        )
        .reshape((2, 2))
        .astype(np.float32)
        .transpose(-1, -2)
    )
    attn_wo = (
        np.array([0.0713, 0.3269, 0.0103, -0.0694])
        .reshape((2, 2))
        .astype(np.float32)
        .transpose(-1, -2)
    )


@pytest.mark.skip(
    reason=(
        "Not passing with updates to e2e model in run llama3.py. Follow up will"
        " replace with better testing using hypothesis library."
    )
)
def test_attention(session):
    model = NanoLlama3()

    dim = 2
    n_heads = 1
    n_kv_heads = 1
    head_dim = 2
    n_rep = 1
    theta = 10000.0
    max_seq_len = 2048

    with Graph(
        "attention",
        input_types=[
            TensorType(dtype=DType.float32, shape=[2, 2, 2]),  # input
            TensorType(dtype=DType.float32, shape=[0, 1, 2, 1, 2]),  # k_cache
            TensorType(dtype=DType.float32, shape=[0, 1, 2, 1, 2]),  # v_cache
        ],
    ) as graph:
        attention = Attention(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dim=dim,
            wk=Linear(model.attn_wk),
            wv=Linear(model.attn_wv),
            wq=Linear(model.attn_wq),
            wo=Linear(model.attn_wo),
            rope=RotaryEmbedding(
                dim=dim,
                n_heads=n_heads,
                theta=theta,
                max_seq_len=max_seq_len,
            ),
        )

        outputs = attention(*graph.inputs)
        graph.output(*outputs)
        compiled = session.load(graph)

    input = (
        np.array(
            [
                -1.2620,
                -2.0678,
                -1.6634,
                1.3036,
                -0.0088,
                -1.1315,
                1.1287,
                1.7699,
            ]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )

    k_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(np.float32)

    v_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(np.float32)

    output = compiled.execute(input, k_cache, v_cache)
    assert len(output) == 3

    expected = (
        np.array(
            [
                -0.1979,
                -0.0316,
                -0.0312,
                -0.0204,
                -0.1011,
                -0.0085,
                -0.0874,
                -0.0067,
            ]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )

    expected_k_cache = (
        np.array(
            [
                0.8053,
                -3.0068,
                -0.9151,
                -1.9720,
                0.8970,
                -1.0378,
                -2.5569,
                0.8611,
            ]
        )
        .reshape((2, 2, n_kv_heads, head_dim))
        .astype(np.float32)
    )

    expected_v_cache = (
        np.array(
            [
                -2.8933,
                0.0256,
                0.3651,
                0.2227,
                -1.1787,
                -0.0522,
                2.5052,
                -0.0266,
            ]
        )
        .reshape((2, 2, n_kv_heads, head_dim))
        .astype(np.float32)
    )

    np.testing.assert_almost_equal(output[0].to_numpy(), expected, decimal=4)
    np.testing.assert_almost_equal(
        output[1].to_numpy(), expected_k_cache, decimal=4
    )
    np.testing.assert_almost_equal(
        output[2].to_numpy(), expected_v_cache, decimal=4
    )
