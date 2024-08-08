# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Temporary Tests to ensure layers execute."""

import max.engine as me
import numpy as np
from max.graph import DType, Graph, TensorType
from dataclasses import dataclass
from llama3.mlp import MLP, Linear
from llama3.attention import Attention
from llama3.transformer import TransformerBlock
from llama3.norm import RMSNorm
from llama3.rotary_embedding import RotaryEmbedding


def test_transformer():
    dim = 2
    n_layers = 1
    n_heads = 1
    vocab_size = 4
    norm_eps = 1e-5
    n_kv_heads = 1
    head_dim = 2
    n_rep = 1
    theta = 10000.0
    max_seq_len = 2048

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
    batch = "batch"
    seq_len = "seq_len"
    prev_seq_len = "prev_seq_len"

    session = me.InferenceSession()

    with Graph(
        "transformer_block",
        input_types=[
            TensorType(dtype=DType.float32, shape=[batch, seq_len, dim]),
            TensorType(
                dtype=DType.float32,
                shape=[prev_seq_len, 1, batch, n_kv_heads, head_dim],
            ),
            TensorType(
                dtype=DType.float32,
                shape=[prev_seq_len, 1, batch, n_kv_heads, head_dim],
            ),
        ],
    ) as graph:
        transformer_block = TransformerBlock(
            attention=Attention(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dim=dim,
                wk=Linear(attn_wk),
                wv=Linear(attn_wv),
                wq=Linear(attn_wq),
                wo=Linear(attn_wo),
                rope=RotaryEmbedding(
                    dim=dim,
                    n_heads=n_heads,
                    theta=theta,
                    max_seq_len=max_seq_len,
                ),
            ),
            mlp=MLP(mlp_w1, mlp_w2, mlp_w3),
            attention_norm=RMSNorm(np.array([-0.0766, 0.6322])),
            mlp_norm=RMSNorm(np.array([-1.0754, -1.1960])),
        )

        # TODO(MSDK-759): Re-enable tests when debugged.
        # graph.output(transformer_block(*graph.inputs))
        # compiled = session.load(graph)

        x = (
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
            .reshape(2, 2, dim)
            .astype(np.float32)
        )

        freqs_cis = (
            np.array([1.0000, 0.0000, 0.5403, 0.8415])
            .reshape(2, 1, 2)
            .astype(np.float32)
        )

        k_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(
            np.float32
        )
        v_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(
            np.float32
        )

        # output = compiled.execute(
        # input0=x, input1=freqs_cis, input2=k_cache, input3=v_cache
        # )

        expected_tokens = (
            np.array(
                [
                    -1.1102,
                    -2.3339,
                    -1.8064,
                    1.4070,
                    0.3818,
                    -1.6129,
                    1.2590,
                    1.6723,
                ]
            )
            .reshape(2, 2, dim)
            .astype(np.float32)
        )

        expected_k_cache = (
            np.array(
                [
                    0.6468,
                    -0.6444,
                    -0.6933,
                    -0.0100,
                    0.7140,
                    -0.8131,
                    -0.8799,
                    -0.1963,
                ]
            )
            .reshape(2, 2, n_kv_heads, head_dim)
            .astype(np.float32)
        )

        expected_v_cache = (
            np.array(
                [
                    -0.7580,
                    -0.0413,
                    0.6225,
                    0.0176,
                    -0.9267,
                    -0.0420,
                    0.7472,
                    0.0410,
                ]
            )
            .reshape(2, 2, n_kv_heads, head_dim)
            .astype(np.float32)
        )

        # np.testing.assert_almost_equal(output["output0"], expected_tokens, decimal=4)
        # np.testing.assert_almost_equal(output["output1"], expected_k_cache, decimal=4)
        # np.testing.assert_almost_equal(output["output2"], expected_v_cache, decimal=4)
