# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.module_v3.rope."""

from __future__ import annotations

from max.nn.module_v3.rope import yarn


def test_yarn() -> None:
    dim = 64
    max_sequence_length = 4096 * 32
    embedding = yarn.positional_embedding(
        dim=dim,
        base=150000,
        alpha=32,
        beta=1,
        max_sequence_length=max_sequence_length,
        original_max_sequence_length=4096,
    )
    assert embedding.shape == [max_sequence_length, dim // 2, 2]
