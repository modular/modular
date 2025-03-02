# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

from builtin._location import __call_location
from max.engine import InferenceSession, TensorMap
from max.tensor import Tensor, TensorShape
from testing import assert_almost_equal, assert_equal


@always_inline
fn assert_tensors_almost_equal[
    dtype: DType
](
    a: Tensor[dtype],
    b: Tensor[dtype],
    atol: Float64 = 1e-8,
    rtol: Float64 = 1e-5,
) raises:
    assert_equal(a.spec(), b.spec())
    for i in range(a.num_elements()):
        assert_almost_equal(
            a.unsafe_ptr()[i],
            b.unsafe_ptr()[i],
            atol=atol,
            rtol=rtol,
            location=__call_location(),
        )


@always_inline
fn assert_tensors_equal[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises:
    assert_equal(a.spec(), b.spec(), location=__call_location())
    for i in range(a.num_elements()):
        assert_equal[dtype](
            a.unsafe_ptr()[i], b.unsafe_ptr()[i], location=__call_location()
        )


@always_inline
fn execute_nullary[
    outtype: DType = DType.float32
](
    graph: Graph,
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> Tensor[outtype]:
    var result_map = execute_no_args(graph, custom_ops_paths=custom_ops_paths)
    return result_map.get[outtype]("output0")


@always_inline
fn execute_nullary_list[
    outtype: DType = DType.float32
](
    graph: Graph,
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> List[
    Tensor[outtype]
]:
    var result_map = execute_no_args(graph, custom_ops_paths=custom_ops_paths)
    var engine_list = result_map.get_value("output0").as_list()
    var results = List[Tensor[outtype]]()
    for i in range(len(engine_list)):
        results.append(engine_list[i].as_tensor_copy[outtype]())
    _ = result_map^
    return results


@always_inline
fn execute_unary[
    intype: DType = DType.float32, outtype: DType = DType.float32
](
    graph: Graph,
    input: Tensor[intype],
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> Tensor[outtype]:
    var result_map = execute_base(
        graph, input, custom_ops_paths=custom_ops_paths
    )
    return result_map.get[outtype]("output0")


@always_inline
fn execute_unary_list[
    intype: DType = DType.float32, outtype: DType = DType.float32
](
    graph: Graph,
    input: Tensor[intype],
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> List[Tensor[outtype]]:
    var result_map = execute_base(
        graph, input, custom_ops_paths=custom_ops_paths
    )
    var engine_list = result_map.get_value("output0").as_list()
    var results = List[Tensor[outtype]]()
    for i in range(len(engine_list)):
        results.append(engine_list[i].as_tensor_copy[outtype]())
    _ = result_map^
    return results


@always_inline
fn execute_binary[
    intype1: DType = DType.float32,
    intype2: DType = intype1,
    outtype: DType = intype1,
](
    graph: Graph,
    x: Tensor[intype1],
    y: Tensor[intype2],
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> Tensor[outtype]:
    graph.verify()

    var session = InferenceSession()
    var model = session.load(graph, custom_ops_paths=custom_ops_paths)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", x)
    input_map.borrow("input1", y)

    var result_map = model.execute(input_map)

    return result_map.get[outtype]("output0")


@always_inline
fn execute_no_args(
    g: Graph,
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> TensorMap:
    g.verify()

    var session = InferenceSession()
    var model = session.load(g, custom_ops_paths=custom_ops_paths)

    var input_map = session.new_tensor_map()
    var result_map = model.execute(input_map)

    return result_map^


@always_inline
fn execute_n_args[
    dt1: DType, dt2: DType, dt3: DType
](
    g: Graph,
    t1: Tensor[dt1],
    t2: Tensor[dt2],
    t3: Tensor[dt3],
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> TensorMap:
    g.verify()

    var session = InferenceSession()
    var model = session.load(g, custom_ops_paths=custom_ops_paths)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", t1)
    input_map.borrow("input1", t2)
    input_map.borrow("input2", t3)

    var result_map = model.execute(input_map)

    return result_map^


@always_inline
fn execute_n_args[
    dt1: DType, dt2: DType, dt3: DType, dt4: DType
](
    g: Graph,
    t1: Tensor[dt1],
    t2: Tensor[dt2],
    t3: Tensor[dt3],
    t4: Tensor[dt4],
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> TensorMap:
    g.verify()

    var session = InferenceSession()
    var model = session.load(g, custom_ops_paths=custom_ops_paths)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", t1)
    input_map.borrow("input1", t2)
    input_map.borrow("input2", t3)
    input_map.borrow("input3", t4)

    var result_map = model.execute(input_map)

    return result_map^


@always_inline
fn execute_n_args[
    dt1: DType, dt2: DType, dt3: DType, dt4: DType, dt5: DType
](
    g: Graph,
    t1: Tensor[dt1],
    t2: Tensor[dt2],
    t3: Tensor[dt3],
    t4: Tensor[dt4],
    t5: Tensor[dt5],
    *,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> TensorMap:
    g.verify()

    var session = InferenceSession()
    var model = session.load(g, custom_ops_paths=custom_ops_paths)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", t1)
    input_map.borrow("input1", t2)
    input_map.borrow("input2", t3)
    input_map.borrow("input3", t4)
    input_map.borrow("input4", t5)

    var result_map = model.execute(input_map)

    return result_map^


@always_inline
fn execute_base(
    g: Graph,
    *tensors: Tensor,
    custom_ops_paths: List[Path] = List[Path](),
) raises -> TensorMap:
    g.verify()

    var session = InferenceSession()
    var model = session.load(g, custom_ops_paths=custom_ops_paths)

    var input_map = session.new_tensor_map()
    for i in range(len(tensors)):
        input_map.borrow(String("input", i), tensors[i])

    var result_map = model.execute(input_map)

    return result_map^
