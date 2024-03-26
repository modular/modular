# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from tensor import Tensor
from testing import assert_almost_equal, assert_equal

from max.engine import InferenceSession, LoadOptions, TensorMap


fn assert_tensors_almost_equal[
    dtype: DType
](
    a: Tensor[dtype],
    b: Tensor[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-5,
) raises:
    assert_equal(a.spec(), b.spec())
    for i in range(a.num_elements()):
        assert_almost_equal[dtype](a[i], b[i], atol=atol, rtol=rtol)


fn assert_tensors_equal[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises:
    assert_equal(a.spec(), b.spec())
    for i in range(a.num_elements()):
        assert_equal[dtype](a[i], b[i])


fn execute_nullary[
    outtype: DType = DType.float32
](module: Module, load_options: Optional[LoadOptions] = None) raises -> Tensor[
    outtype
]:
    var result_map = execute_no_args(module, load_options)
    return result_map.get[outtype]("output0")


fn execute_nullary_list[
    outtype: DType = DType.float32
](module: Module, load_options: Optional[LoadOptions] = None) raises -> List[
    Tensor[outtype]
]:
    var result_map = execute_no_args(module, load_options)
    var engine_list = result_map.get_value("output0").as_list()
    var results = List[Tensor[outtype]]()
    for i in range(len(engine_list)):
        results.append(engine_list[i].as_tensor_copy[outtype]())
    _ = result_map^
    return results


fn execute_unary[
    intype: DType = DType.float32, outtype: DType = DType.float32
](
    module: Module,
    input: Tensor[intype],
    load_options: Optional[LoadOptions] = None,
) raises -> Tensor[outtype]:
    var result_map = execute_base(module, input, load_options=load_options)
    return result_map.get[outtype]("output0")


fn execute_unary_list[
    intype: DType = DType.float32, outtype: DType = DType.float32
](
    module: Module,
    input: Tensor[intype],
    load_options: Optional[LoadOptions] = None,
) raises -> List[Tensor[outtype]]:
    var result_map = execute_base(module, input, load_options=load_options)
    var engine_list = result_map.get_value("output0").as_list()
    var results = List[Tensor[outtype]]()
    for i in range(len(engine_list)):
        results.append(engine_list[i].as_tensor_copy[outtype]())
    _ = result_map^
    return results


fn execute_binary[
    intype: DType = DType.float32, outtype: DType = DType.float32
](
    module: Module,
    x: Tensor[intype],
    y: Tensor[intype],
    load_options: Optional[LoadOptions] = None,
) raises -> Tensor[outtype]:
    var result_map = execute_base(module, x, y, load_options=load_options)
    return result_map.get[outtype]("output0")


fn execute_no_args(
    m: Module, load_options: Optional[LoadOptions] = None
) raises -> TensorMap:
    m.verify()

    var session = InferenceSession()
    var model = session.load_model(m, load_options)

    var input_map = session.new_tensor_map()
    var result_map = model.execute(input_map)

    return result_map^


fn execute_n_args[
    dt1: DType, dt2: DType, dt3: DType, dt4: DType, dt5: DType
](
    m: Module,
    t1: Tensor[dt1],
    t2: Tensor[dt2],
    t3: Tensor[dt3],
    t4: Tensor[dt4],
    t5: Tensor[dt5],
    load_options: Optional[LoadOptions] = None,
) raises -> TensorMap:
    m.verify()

    var session = InferenceSession()
    var model = session.load_model(m, load_options)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", t1)
    input_map.borrow("input1", t2)
    input_map.borrow("input2", t3)
    input_map.borrow("input3", t4)
    input_map.borrow("input4", t5)

    var result_map = model.execute(input_map)

    return result_map^


fn execute_base(
    m: Module, *tensors: Tensor, load_options: Optional[LoadOptions] = None
) raises -> TensorMap:
    m.verify()

    var session = InferenceSession()
    var model = session.load_model(m, load_options)

    var input_map = session.new_tensor_map()
    for i in range(len(tensors)):
        input_map.borrow("input" + str(i), tensors[i])

    var result_map = model.execute(input_map)

    return result_map^
