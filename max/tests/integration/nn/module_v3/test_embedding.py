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
"""Tests for max.nn.Linear."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding


def test_repr() -> None:
    assert (
        repr(Embedding(2, dim=2)) == "Embedding(vocab_size=Dim(2), dim=Dim(2))"
    )
    assert (
        repr(Embedding(2, dims=(2, 3)))
        == "Embedding(vocab_size=Dim(2), dims=[Dim(2), Dim(3)])"
    )


def test_vocab_size() -> None:
    assert Embedding(2, dim=3).vocab_size == 2


def test_dim() -> None:
    assert Embedding(2, dim=3).dim == 3

    with pytest.raises(TypeError):
        _ = Embedding(2, dims=(3, 4)).dim


def test_dims() -> None:
    assert Embedding(2, dim=3).dims == [3]
    assert Embedding(2, dims=(3, 4)).dims == [3, 4]


def test_parameters() -> None:
    embedding = Embedding(2, dim=3)
    assert dict(embedding.parameters) == {"weight": embedding.weight}


def test_call() -> None:
    embedding = Embedding(2, dim=3)
    result = embedding(Tensor.ones([10], dtype=DType.uint64))
    assert result.shape == [10, 3]

    result = embedding(Tensor.ones([5, 10], dtype=DType.uint64))
    assert result.shape == [5, 10, 3]

    two_d_embedding = Embedding(2, dims=(3, 4))
    result = two_d_embedding(Tensor.ones([10], dtype=DType.uint64))
    assert result.shape == [10, 3, 4]

    result = two_d_embedding(Tensor.ones([5, 10], dtype=DType.uint64))
    assert result.shape == [5, 10, 3, 4]
