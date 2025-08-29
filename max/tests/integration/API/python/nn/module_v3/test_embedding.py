# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.module_v3.Linear."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding


def test_repr():
    assert (
        repr(Embedding(2, dim=2)) == "Embedding(vocab_size=Dim(2), dim=Dim(2))"
    )
    assert (
        repr(Embedding(2, dims=(2, 3)))
        == "Embedding(vocab_size=Dim(2), dims=[Dim(2), Dim(3)])"
    )


def test_vocab_size():
    assert Embedding(2, dim=3).vocab_size == 2


def test_dim():
    assert Embedding(2, dim=3).dim == 3

    with pytest.raises(TypeError):
        _ = Embedding(2, dims=(3, 4)).dim


def test_dims():
    assert Embedding(2, dim=3).dims == [3]
    assert Embedding(2, dims=(3, 4)).dims == [3, 4]


def test_parameters():
    embedding = Embedding(2, dim=3)
    assert dict(embedding.parameters) == {"weight": embedding.weight}


def test_call():
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
