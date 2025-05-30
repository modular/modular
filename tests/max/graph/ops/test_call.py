# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.call tests."""

import re

import pytest
from conftest import tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    Weight,
    _ChainType,
    _ChainValue,
    ops,
)
from max.mlir.dialects import mo

# Create shared strategies for tensor types
input_types = st.shared(tensor_types())


def create_simple_subgraph(graph: Graph, input_type: TensorType) -> Graph:
    """Creates a simple graph that adds 1 to its input."""
    with graph.add_subgraph("add_one", input_types=[input_type]) as subgraph:
        x = subgraph.inputs[0].tensor
        one = ops.constant(1, input_type.dtype, device=DeviceRef.CPU())
        out = ops.elementwise.add(x, one)
        subgraph.output(out)
    return subgraph


def create_multi_input_subgraph(
    graph: Graph, input_types: list[TensorType]
) -> Graph:
    """Creates a graph that adds its inputs together."""
    with graph.add_subgraph("add_inputs", input_types=input_types) as subgraph:
        # Add inputs sequentially instead of using sum()
        result = subgraph.inputs[0].tensor
        for x in subgraph.inputs[1:]:
            if result.dtype == DType.bool:
                result = ops.elementwise.logical_and(result, x.tensor)
            else:
                result = ops.elementwise.add(result, x.tensor)
        subgraph.output(result)
    return subgraph


def create_multi_output_subgraph(graph: Graph, input_type: TensorType) -> Graph:
    """Creates a graph that returns multiple outputs."""
    with graph.add_subgraph(
        "multi_output", input_types=[input_type]
    ) as subgraph:
        x = subgraph.inputs[0].tensor
        if x.dtype == DType.bool:
            x = ops.cast(x, DType.int8)
        one = ops.constant(1, x.dtype, device=DeviceRef.CPU())
        two = ops.constant(2, x.dtype, device=DeviceRef.CPU())
        out1 = ops.elementwise.add(x, one)
        out2 = ops.elementwise.mul(x, two)
        if input_type.dtype == DType.bool:
            out1 = ops.cast(out1, DType.bool)
            out2 = ops.cast(out2, DType.bool)
        subgraph.output(out1, out2)
    return subgraph


@given(input_type=tensor_types())
def test_call_simple_graph(input_type: TensorType):
    """Test calling a simple graph with a single input and output."""

    with Graph(
        "main",
        input_types=[input_type],
    ) as main_graph:
        subgraph = create_simple_subgraph(main_graph, input_type)
        result = ops.call(subgraph, main_graph.inputs[0])
        assert len(result) == 1
        assert result[0].type == input_type


@given(input_type=tensor_types())
def test_call_multi_output(input_type: TensorType):
    """Test calling a graph that returns multiple outputs."""

    with Graph(
        "main",
        input_types=[input_type],
    ) as main_graph:
        subgraph = create_multi_output_subgraph(main_graph, input_type)
        results = ops.call(subgraph, main_graph.inputs[0])
        assert len(results) == 2
        assert all(r.type == input_type for r in results)


@given(input_type=tensor_types())
def test_call_nested(input_type: TensorType):
    """Test nested graph calls."""

    with Graph(
        "outer",
        input_types=[input_type],
    ) as outer_graph:
        with outer_graph.add_subgraph(
            "middle",
            input_types=[input_type],
        ) as middle_graph:
            inner_graph = create_simple_subgraph(middle_graph, input_type)
            x = middle_graph.inputs[0]
            y = ops.call(inner_graph, x)[0]
            middle_graph.output(y)
        result = ops.call(middle_graph, outer_graph.inputs[0])
        assert len(result) == 1
        assert result[0].type == input_type


def test_call_type_mismatch():
    """Test that calling a graph with mismatched types raises an error."""
    float_type = TensorType(DType.float32, [10], DeviceRef.CPU())
    int_type = TensorType(DType.int32, [10], DeviceRef.CPU())

    with Graph(
        "main",
        input_types=[int_type],
    ) as main_graph:
        subgraph = create_simple_subgraph(main_graph, float_type)
        with pytest.raises(ValueError, match="wrong type"):
            ops.call(subgraph, main_graph.inputs[0])


@given(input_type=tensor_types())
def test_call_multi_input(input_type: TensorType):
    """Test calling a graph with multiple inputs."""
    input_types = [input_type] * 4

    with Graph(
        "main",
        input_types=input_types,
    ) as main_graph:
        subgraph = create_multi_input_subgraph(main_graph, input_types)
        results = ops.call(subgraph, *main_graph.inputs)
        assert len(results) == 1
        # The output type should match the first input type when adding tensors
        assert results[0].type == input_type


def test_call_num_inputs_mismatch():
    """Test calling a graph with a mismatch in the number of inputs."""
    input_types = [TensorType(DType.float32, [4], DeviceRef.CPU())] * 4

    with Graph(
        "main",
        input_types=input_types,
    ) as main_graph:
        subgraph = create_multi_input_subgraph(main_graph, input_types)
        with pytest.raises(ValueError, match="Expected 4 args.*, got 1"):
            ops.call(subgraph, main_graph.inputs[0])


def test_call_chain_updates():
    """Test that calling a subgraph with chain input/output updates the chain."""
    buffer_type = BufferType(DType.float32, [4], DeviceRef.CPU())
    tensor_type = TensorType(DType.float32, [4], DeviceRef.CPU())

    with Graph("main", input_types=[buffer_type, tensor_type]) as main_graph:
        # Subgraph that stores tensor into buffer (mutates state, uses chain)
        with main_graph.add_subgraph(
            "store_subgraph",
            input_types=[buffer_type, tensor_type, _ChainType()],
        ) as subgraph:
            buffer = subgraph.inputs[0]
            tensor = subgraph.inputs[1]
            buf_val = buffer.buffer
            ten_val = tensor.tensor
            ops.buffer_store(buf_val, ten_val)
            subgraph.output(subgraph._current_chain)
        buffer = main_graph.inputs[0]
        tensor = main_graph.inputs[1]
        chain_before = main_graph._current_chain
        ops.call(subgraph, buffer, tensor)
        chain_after = main_graph._current_chain
        assert isinstance(chain_before, _ChainValue)
        assert isinstance(chain_after, _ChainValue)
        assert chain_before != chain_after


def test_call_chain_input_output_mismatch():
    """Test that a subgraph with chain input but no chain output (or vice versa) raises ValueError."""
    buffer_type = BufferType(DType.float32, [4], DeviceRef.CPU())
    tensor_type = TensorType(DType.float32, [4], DeviceRef.CPU())
    with Graph("main", input_types=[buffer_type, tensor_type]) as main_graph:
        # Manually create a subgraph with chain input but not outputting it
        with main_graph.add_subgraph(
            "bad_chain_subgraph",
            input_types=[buffer_type, tensor_type, _ChainType()],
        ) as subgraph:
            buffer = subgraph.inputs[0]
            tensor = subgraph.inputs[1]
            buf_val = buffer.buffer
            ten_val = tensor.tensor
            # This will use the chain, but we intentionally do NOT output anything (no chain output)
            ops.buffer_store(buf_val, ten_val)
            subgraph.output()  # no chain!
        buffer = main_graph.inputs[0]
        tensor = main_graph.inputs[1]
        with pytest.raises(ValueError, match="must have.*chain output"):
            ops.call(subgraph, buffer, tensor)


def test_call_no_chain_no_update():
    """Test that calling a subgraph with no chain input/output does not update the chain."""
    tensor_type = TensorType(DType.float32, [4], DeviceRef.CPU())
    with Graph("main", input_types=[tensor_type]) as main_graph:
        with main_graph.add_subgraph(
            "add_one", input_types=[tensor_type]
        ) as subgraph:
            x = subgraph.inputs[0].tensor
            out = ops.elementwise.add(
                x, ops.constant(1, DType.float32, device=DeviceRef.CPU())
            )
            subgraph.output(out)
        x = main_graph.inputs[0]
        chain_before = main_graph._current_chain
        ops.call(subgraph, x)
        chain_after = main_graph._current_chain
        assert chain_before == chain_after


def test_call_tuple_operands_with_add_op():
    """Test calling a graph using _add_op with tuple for operands."""
    input_type = TensorType(DType.float32, [10], DeviceRef.CPU())
    with Graph("main_graph_tuple_test", input_types=[input_type]) as main_graph:
        # Create a simple subgraph that just returns its input.
        with main_graph.add_subgraph(
            "identity_subgraph", input_types=[input_type]
        ) as subgraph:
            subgraph.output(subgraph.inputs[0])

        # Call the subgraph using _add_op with operands as a tuple.
        # This is the core of the test: ensuring unwrap handles the tuple.
        call_results = main_graph._add_op(
            mo.call_,
            symbol=subgraph.name,
            results=(input_type,),
            operands=(main_graph.inputs[0],),
        )

        assert len(call_results) == 1
        assert call_results[0].type == input_type
        main_graph.output(call_results[0])


def test_call_with_prefix():
    """Test calling a graph with a prefix of a subgraph that has a placeholder weight."""
    input_type = TensorType(DType.float32, [10], DeviceRef.CPU())
    with Graph(
        "main_graph_prefix_test", input_types=[input_type]
    ) as main_graph:
        with main_graph.add_subgraph(
            "subgraph", input_types=[input_type]
        ) as subgraph:
            w = Weight(
                "placeholder",
                dtype=DType.float32,
                shape=[10],
                device=DeviceRef.CPU(),
                _placeholder=True,
            )
            subgraph.output(subgraph.inputs[0] + w)
        assert re.search(
            r"mo.constant.external.*isPlaceholder = true.*!mo.tensor<\[10\], f32",
            str(subgraph),
        )
        call_results = ops.call(subgraph, main_graph.inputs[0], prefix="prefix")
        assert len(call_results) == 1
        assert call_results[0].type == input_type
        main_graph.output(call_results[0])
    assert re.search(
        r"mo.call @subgraph.*\{prefix = \"prefix\"\}.*!mo.tensor<\[10\], f32",
        str(main_graph),
    )
