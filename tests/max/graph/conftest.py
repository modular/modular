# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from hypothesis import strategies as st
from max import _graph, mlir
from max.graph import DType, TensorType
from max.graph.type import Dim, StaticDim, SymbolicDim

dtypes = st.sampled_from(DType)
static_dims = st.builds(
    StaticDim, st.integers(min_value=0, max_value=2**63 - 1)
)
symbolic_dims = st.builds(
    SymbolicDim,
    st.one_of(
        st.just("batch"),
        st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    ),
)
dims = st.one_of(static_dims, symbolic_dims)
tensor_types = st.builds(TensorType, dtypes, st.lists(dims))

st.register_type_strategy(DType, dtypes)
st.register_type_strategy(Dim, dims)
st.register_type_strategy(StaticDim, static_dims)
st.register_type_strategy(SymbolicDim, symbolic_dims)
st.register_type_strategy(TensorType, tensor_types)


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture(scope="function")
def mlir_context() -> mlir.Context:
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        registry = mlir.DialectRegistry()
        _graph.load_modular_dialects(registry)
        ctx.append_dialect_registry(registry)
        ctx.load_all_available_dialects()
        yield ctx
