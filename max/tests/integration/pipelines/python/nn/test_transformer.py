# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Temporary Tests to ensure layers execute."""

from dataclasses import dataclass

import max.engine as me
import numpy as np
import pytest
from max.dtype import DType
from max.graph import Graph, TensorType
from nn import (
    MLP,
    NaiveAttentionWithRope,
    Embedding,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)


class Weights:
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
    ).transpose(-1, -2)
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
    ).transpose(-1, -2)
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
    ).transpose(-1, -2)
    attn_wo = (
        np.array([0.0713, 0.3269, 0.0103, -0.0694])
        .reshape((2, 2))
        .astype(np.float32)
    ).transpose(-1, -2)

    mlp_w1 = np.array(
        [
            [0.5641, 0.4875],
            [-1.1172, -1.1583],
        ]
    ).transpose(-1, -2)
    mlp_w2 = np.array(
        [
            [0.5355, -0.9487],
            [-0.6487, 0.1838],
        ]
    ).transpose(-1, -2)
    mlp_w3 = np.array(
        [
            [-0.6765, 0.7103],
            [-0.4643, 0.2860],
        ]
    ).transpose(-1, -2)
    output_weight = (
        np.array(
            [
                0.1539,
                0.0616,
                0.5123,
                -0.3383,
                0.3272,
                0.9645,
                -0.7428,
                -0.1215,
            ]
        )
        .astype(np.float32)
        .reshape([4, 2])
    ).transpose()
    token_embedding = (
        np.array(
            [
                0.7091,
                -0.6393,
                -1.0965,
                -0.0201,
                -0.3484,
                0.0024,
                -2.0185,
                -0.4979,
            ]
        )
        .astype(np.float32)
        .reshape(4, 2)
    )


@pytest.mark.skip(
    reason=(
        "Not passing with updates to e2e model in run llama3.py. Follow up will"
        " replace with better testing using hypothesis library."
    )
)
def test_transformer_block(session):
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

    batch = "batch"
    seq_len = "seq_len"
    prev_seq_len = "prev_seq_len"

    w = Weights()

    with Graph(
        "transformer_block",
        input_types=[
            TensorType(dtype=DType.float32, shape=[2, 2, 2]),
            TensorType(
                dtype=DType.float32,
                shape=[0, 1, 2, n_kv_heads, head_dim],
            ),
            TensorType(
                dtype=DType.float32,
                shape=[0, 1, 2, n_kv_heads, head_dim],
            ),
        ],
    ) as graph:
        transformer_block = TransformerBlock(
            attention=NaiveAttentionWithRope(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dim=dim,
                wk=Linear(w.attn_wk),
                wv=Linear(w.attn_wv),
                wq=Linear(w.attn_wq),
                wo=Linear(w.attn_wo),
                rope=RotaryEmbedding(
                    dim=dim,
                    n_heads=n_heads,
                    theta=theta,
                    max_seq_len=max_seq_len,
                ),
            ),
            mlp=MLP(w.mlp_w1, w.mlp_w2, w.mlp_w3),
            attention_norm=RMSNorm(np.array([-0.0766, 0.6322])),
            mlp_norm=RMSNorm(np.array([-1.0754, -1.1960])),
        )

        outputs = transformer_block(*graph.inputs)
        graph.output(*outputs)
        compiled = session.load(graph)

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

        k_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(
            np.float32
        )
        v_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(
            np.float32
        )

        output = compiled.execute(x, k_cache, v_cache)
        assert len(output) == 3

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

        np.testing.assert_almost_equal(
            output[0].to_numpy(), expected_tokens, decimal=4
        )
        np.testing.assert_almost_equal(
            output[1].to_numpy(), expected_k_cache, decimal=4
        )
        np.testing.assert_almost_equal(
            output[2].to_numpy(), expected_v_cache, decimal=4
        )


@pytest.mark.skip(
    reason=(
        "Not passing with updates to e2e model in run llama3.py. Follow up will"
        " replace with better testing using hypothesis library."
    )
)
def test_transformer():
    dim = 2
    n_heads = 1
    n_kv_heads = 1
    head_dim = 2
    theta = 10000.0
    max_seq_len = 2048
    rope_scaling = None

    w = Weights()

    session = me.InferenceSession()

    with Graph(
        "transformer",
        input_types=[
            TensorType(dtype=DType.int64, shape=[2, 2]),
            TensorType(
                dtype=DType.float32,
                shape=[0, 1, 2, n_kv_heads, head_dim],
            ),
            TensorType(
                dtype=DType.float32,
                shape=[0, 1, 2, n_kv_heads, head_dim],
            ),
        ],
    ) as graph:
        transformer = Transformer(
            dim=dim,
            n_heads=n_heads,
            layers=[
                TransformerBlock(
                    attention=NaiveAttentionWithRope(
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        dim=dim,
                        wk=Linear(w.attn_wk),
                        wv=Linear(w.attn_wv),
                        wq=Linear(w.attn_wq),
                        wo=Linear(w.attn_wo),
                        rope=RotaryEmbedding(
                            dim=dim,
                            n_heads=n_heads,
                            theta=theta,
                            max_seq_len=max_seq_len,
                        ),
                    ),
                    mlp=MLP(w.mlp_w1, w.mlp_w2, w.mlp_w3),
                    attention_norm=RMSNorm(np.array([-0.0766, 0.6322])),
                    mlp_norm=RMSNorm(np.array([-1.0754, -1.1960])),
                )
            ],
            norm=RMSNorm(np.array([1.0476, -0.3264])),
            output=Linear(w.output_weight),
            theta=theta,
            embedding=Embedding(w.token_embedding),
            rope_scaling=rope_scaling,
        )
        graph.output(*transformer(*graph.inputs))
        compiled = session.load(graph)

        tokens = np.array([2, 0, 1, 2]).astype(np.int64).reshape([2, 2])

        k_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(
            np.float32
        )
        v_cache = np.zeros(shape=(0, 1, 2, n_kv_heads, head_dim)).astype(
            np.float32
        )
        output = compiled.execute(tokens, k_cache, v_cache)
        assert len(output) == 3

        expected_tokens = (
            np.array(
                [
                    -0.2158,
                    -0.6037,
                    -0.6346,
                    1.0045,
                    0.1893,
                    0.4635,
                    0.6581,
                    -0.8597,
                    -0.2278,
                    -0.6952,
                    -0.5812,
                    1.0791,
                    -0.2158,
                    -0.6040,
                    -0.6345,
                    1.0047,
                ]
            )
            .reshape(2, 2, 4)
            .astype(np.float32)
        )

        expected_k_cache = (
            np.array(
                [
                    0.0676,
                    0.1021,
                    0.0856,
                    0.0816,
                    0.7479,
                    0.0235,
                    -0.0494,
                    0.1121,
                ]
            )
            .reshape(2, 1, 2, 1, 2)
            .astype(np.float32)
        )

        expected_v_cache = (
            np.array(
                [
                    0.0707,
                    -0.0102,
                    0.0473,
                    -0.0113,
                    -0.6686,
                    -0.0203,
                    0.0707,
                    -0.0102,
                ]
            )
            .reshape(2, 1, 2, 1, 2)
            .astype(np.float32)
        )

        np.testing.assert_almost_equal(
            output[0].to_numpy(), expected_tokens, decimal=4
        )
        np.testing.assert_almost_equal(
            output[1].to_numpy(), expected_k_cache, decimal=4
        )
        np.testing.assert_almost_equal(
            output[2].to_numpy(), expected_v_cache, decimal=4
        )
