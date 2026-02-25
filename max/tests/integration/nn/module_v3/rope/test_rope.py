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
"""Tests for max.nn.rope."""

from __future__ import annotations

import pytest
from max.experimental.tensor import Tensor, TensorType
from max.nn.module_v3.rope import (
    RotaryEmbedding,
    TransposedRotaryEmbedding,
    rope,
)


def test_repr() -> None:
    assert (
        repr(RotaryEmbedding(Tensor.zeros([2, 3, 2])))
        == "RotaryEmbedding(dim=6, max_sequence_length=2)"
    )


def test_dim() -> None:
    assert RotaryEmbedding(Tensor.zeros([2, 3, 2])).dim == 6


def test_max_sequence_length() -> None:
    assert RotaryEmbedding(Tensor.zeros([2, 3, 2])).max_sequence_length == 2


def test_parameters() -> None:
    embedding = RotaryEmbedding(Tensor.zeros([2, 3, 2]))
    assert dict(embedding.parameters) == {
        "weight": embedding.weight,
    }


def test_call() -> None:
    embedding = RotaryEmbedding(Tensor.zeros([2, 3, 2]))
    result = embedding(Tensor.ones([1, 2, 1, 6]), 0)
    assert result.shape == [1, 2, 1, 6]


def test_transposed_rotary_embedding() -> None:
    embedding = TransposedRotaryEmbedding(Tensor.zeros([2, 3, 2]))
    result = embedding(Tensor.ones([1, 2, 1, 6]), 0)
    assert result.shape == [1, 2, 1, 6]


def test_inverse_exponential_frequencies() -> None:
    freqs = rope.theta(dim=20, base=10000)
    assert freqs.shape == [10]  # [20 // 2]


def test_positional_embedding() -> None:
    freqs = rope.positional_embedding(dim=20, base=10000, max_sequence_length=5)
    assert freqs.shape == [5, 10, 2]  # [5, 20 // 2, 2]


def test_symbolic_seqlen() -> None:
    embedding = RotaryEmbedding(Tensor.zeros([2, 3, 2]))

    compiled = embedding.compile(
        TensorType(
            embedding.weight.dtype,
            ["batch", "seqlen", "n_kv_heads", embedding.dim],
            embedding.weight.device,
        )
    )

    assert compiled(Tensor.zeros([1, 1, 1, 6])).shape == [1, 1, 1, 6]
    assert compiled(Tensor.zeros([3, 2, 5, 6])).shape == [3, 2, 5, 6]

    # TODO(XFN-23): make this fail
    # seqlen > max_sequence_length
    # with pytest.raises(ValueError):
    #     compiled(Tensor.zeros([1, 3, 1, 6]))

    assert 5 != embedding.dim
    with pytest.raises(ValueError):
        compiled(Tensor.zeros([1, 2, 1, 5]))
