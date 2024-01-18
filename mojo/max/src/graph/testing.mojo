# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.param_env import env_get_string
from tensor import Tensor, TensorShape, TensorSpec
from testing import assert_true, assert_almost_equal, assert_equal

from max.engine import EngineTensorView, InferenceSession, TensorMap


alias TEMP_DIR = Path(env_get_string["TEMP_DIR", ""]())


fn assert_no_dynamic_dims(spec: TensorSpec) raises:
    for i in range(spec.rank()):
        assert_true(spec[i] >= 0)


fn assert_tensors_almost_equal[
    dtype: DType
](
    a: Tensor[dtype],
    b: Tensor[dtype],
    atol: Scalar[dtype] = 1e-8,
    rtol: Scalar[dtype] = 1e-5,
) raises:
    assert_true(a.spec() == b.spec())
    assert_no_dynamic_dims(a.spec())
    assert_no_dynamic_dims(b.spec())

    for i in range(a.num_elements()):
        assert_almost_equal[dtype](a[i], b[i], atol, rtol)


fn assert_tensors_equal[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises:
    assert_true(a.spec() == b.spec())
    assert_no_dynamic_dims(a.spec())
    assert_no_dynamic_dims(b.spec())

    for i in range(a.num_elements()):
        assert_equal[dtype](a[i], b[i])


fn execute_nullary[
    outtype: DType = DType.float32
](module: Module, name: StringRef) raises -> Tensor[outtype]:
    let result_map = execute_no_args(module, name)
    return result_map.get[outtype]("output0")


fn execute_unary[
    intype: DType = DType.float32, outtype: DType = DType.float32
](module: Module, name: StringRef, input: Tensor[intype]) raises -> Tensor[
    outtype
]:
    let result_map = execute_base(module, name, input)
    return result_map.get[outtype]("output0")


fn execute_binary[
    intype: DType = DType.float32, outtype: DType = DType.float32
](
    module: Module, name: StringRef, x: Tensor[intype], y: Tensor[intype]
) raises -> Tensor[outtype]:
    let result_map = execute_base(module, name, x, y)
    return result_map.get[outtype]("output0")


fn execute_no_args(m: Module, name: StringRef) raises -> TensorMap:
    m.verify()

    # TODO: Don't ask the user to provide a unique name, autogenerate it.
    let temp_path = TEMP_DIR / name

    with open(temp_path, "w") as f:
        f.write(m.__str__())

    let session = InferenceSession()
    let model = session.load_model(temp_path)

    let input_map = session.new_tensor_map()
    let result_map = model.execute(input_map)
    temp_path.path._strref_keepalive()

    return result_map ^


fn execute_n_args[
    dt1: DType, dt2: DType, dt3: DType, dt4: DType, dt5: DType
](
    m: Module,
    name: StringRef,
    t1: Tensor[dt1],
    t2: Tensor[dt2],
    t3: Tensor[dt3],
    t4: Tensor[dt4],
    t5: Tensor[dt5],
) raises -> TensorMap:
    m.verify()

    # TODO: Don't ask the user to provide a unique name, autogenerate it.
    let temp_path = TEMP_DIR / name

    with open(temp_path, "w") as f:
        f.write(m.__str__())

    let session = InferenceSession()
    let model = session.load_model(temp_path)

    let input_map = session.new_tensor_map()
    input_map.borrow("input0", t1)
    input_map.borrow("input1", t2)
    input_map.borrow("input2", t3)
    input_map.borrow("input3", t4)
    input_map.borrow("input4", t5)

    let result_map = model.execute(input_map)
    temp_path.path._strref_keepalive()

    return result_map ^


fn execute_base(
    m: Module, name: StringRef, *tensors: Tensor
) raises -> TensorMap:
    m.verify()

    # TODO: Don't ask the user to provide a unique name, autogenerate it.
    let temp_path = TEMP_DIR / name

    with open(temp_path, "w") as f:
        f.write(m.__str__())

    let session = InferenceSession()
    let model = session.load_model(temp_path)

    let input_map = session.new_tensor_map()
    for i in range(len(tensors)):
        let input_name = "input" + String(i)
        input_map.borrow(
            input_name._strref_dangerous(),
            tensors[i],
        )
        input_name._strref_keepalive()

    let result_map = model.execute(input_map)
    temp_path.path._strref_keepalive()

    return result_map ^
