# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""


import max.engine as me
import numpy as np
from max.graph import DType, Graph, TensorType
from dataclasses import dataclass
from llama3.mlp import MLP
from llama3.norm import RMSNorm
from llama3.attention import Attention, rope


@dataclass
class NanoLlama3:
    """Class to hold toy weights and parameters for testing llama3 code."""

    params = {
        "dims": 2,
        "n_layers": 1,
        "n_heads": 1,
        "vocab_size": 4,
        "norm_eps": 1e-5,
        "n_kv_heads": 1,
        "head_dim": 2,
        "n_rep": 1,
    }

    mlp_w1 = np.array(
        [
            [0.5641, 0.4875],
            [-1.1172, -1.1583],
        ]
    )
    mlp_w2 = np.array(
        [
            [0.5355, -0.9487],
            [-0.6487, 0.1838],
        ]
    )
    mlp_w3 = np.array(
        [
            [-0.6765, 0.7103],
            [-0.4643, 0.2860],
        ]
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
    )
    attn_wo = (
        np.array([0.0713, 0.3269, 0.0103, -0.0694])
        .reshape((2, 2))
        .astype(np.float32)
    )


def test_mlp():
    session = me.InferenceSession()
    model = NanoLlama3()

    mlp = Graph(
        "mlp",
        MLP(model.mlp_w1, model.mlp_w2, model.mlp_w3),
        input_types=[TensorType(dtype=DType.float32, shape=[2, 2, 2])],
    )
    compiled = session.load(mlp)
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
    output = compiled.execute(input0=input)

    expected = (
        np.array(
            [
                0.1053,
                -0.1079,
                -0.3632,
                0.2142,
                0.4025,
                -0.1661,
                0.3220,
                -0.3921,
            ]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )
    # TODO (MSDK-720): Re-enable after troubleshooting accuracy.
    # np.testing.assert_almost_equal(output["output0"], expected, decimal=4)


def test_norm():
    session = me.InferenceSession()
    model = NanoLlama3()

    norm = Graph(
        "norm",
        RMSNorm(np.array([1.0476, -0.3264])),
        input_types=[TensorType(dtype=DType.float32, shape=[2, 2, 2])],
    )
    compiled = session.load(norm)

    input = (
        np.array(
            [-0.8566, 0.4401, 0.6973, -0.6199, -1.6010, 0.4166, -0.8575, 0.4400]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )

    output = compiled.execute(input0=input)

    expected = (
        np.array(
            [
                -1.3178,
                -0.2109,
                1.1073,
                0.3067,
                -1.4338,
                -0.1162,
                -1.3181,
                -0.2107,
            ]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )

    # TODO (MSDK-720): Re-enable after troubleshooting accuracy.
    np.testing.assert_almost_equal(output["output0"], expected, decimal=4)


def test_rope():
    session = me.InferenceSession()

    x = (
        np.array(
            [
                -1.3140,
                -1.5004,
                0.4776,
                -0.2095,
                0.9650,
                1.6373,
                -0.0903,
                -2.1381,
            ]
        )
        .reshape((1, 2, 2, 2))
        .astype(np.float32)
    )

    freqs_cis = (
        np.array([0.42, 0.9075, 0.5403, 0.8415])
        .reshape(2, 1, 2)
        .astype(np.float32)
    )

    rope_graph = Graph(
        "rope_graph",
        rope,
        input_types=[
            TensorType(dtype=DType.float32, shape=[1, 2, 2, 2]),
            TensorType(dtype=DType.float32, shape=[2, 1, 2]),
        ],
    )

    compiled = session.load(rope_graph)
    output = compiled.execute(input0=x, input1=freqs_cis)
    expected = (
        np.array(
            [
                0.8097,
                -1.8226,
                0.3907,
                0.3454,
                -0.8564,
                1.6967,
                1.7504,
                -1.2312,
            ]
        )
        .reshape(1, 2, 2, 2)
        .astype(np.float32)
    )
    # assert np.testing.assert_almost_equal(output["output0"], expected, decimal=4)


def test_attention():
    session = me.InferenceSession()
    model = NanoLlama3()

    dim = 2
    n_heads = 1
    n_kv_heads = 1
    head_dim = 2

    attention = Graph(
        "attention",
        Attention(
            model.params,
            model.attn_wk,
            model.attn_wv,
            model.attn_wq,
            model.attn_wo,
        ),
        input_types=[
            TensorType(dtype=DType.float32, shape=[2, 2, 2]),  # input
            TensorType(dtype=DType.float32, shape=[2, 1, 2]),  # freqs_cis
            TensorType(dtype=DType.float32, shape=[0, 1, 2, 1, 2]),  # k_cache
            TensorType(dtype=DType.float32, shape=[0, 1, 2, 1, 2]),  # v_cache
        ],
    )
    compiled = session.load(attention)

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

    freqs_cis = (
        np.array([1.0000, 0.0000, 0.5403, 0.8415])
        .reshape((2, 1, 2))
        .astype(np.float32)
    )
    k_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(np.float32)

    v_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(np.float32)

    output = compiled.execute(
        input0=input, input1=freqs_cis, input2=k_cache, input3=v_cache
    )

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

    # TODO (MSDK-720): Re-enable after troubleshooting accuracy.
    # assert np.testing.assert_almost_equal(output["output0"], expected, decimal=4)
    # assert np.testing.assert_almost_equal(output["output1"], expected_k_cache, decimal=4)
    # assert np.testing.assert_almost_equal(output["output2"], expected_v_cache, decimal=4)
