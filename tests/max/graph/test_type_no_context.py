# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests type factoriy and accessor errors for when an MLIR context has not
been initialized."""

import pytest
from hypothesis import given
from max.graph import (
    BufferType,
    Dim,
    TensorType,
    _ChainType,
    _OpaqueType,
)


@given(dim=...)
def test_dim_to_mlir_no_context(dim: Dim):
    with pytest.raises(RuntimeError):
        print(dim.to_mlir())


@given(tensor_type=...)
def test_tensor_type_to_mlir_no_context(tensor_type: TensorType):
    with pytest.raises(RuntimeError):
        tensor_type.to_mlir()


def test_opaque_type_to_mlir_no_context():
    with pytest.raises(RuntimeError):
        _OpaqueType("something").to_mlir()


@given(buffer_type=...)
def test_buffer_mlir_roundtrip_no_context(buffer_type: BufferType):
    with pytest.raises(RuntimeError):
        buffer_type.to_mlir()


def test_chain_type__no_mlir_context():
    with pytest.raises(RuntimeError):
        _ChainType().to_mlir()
