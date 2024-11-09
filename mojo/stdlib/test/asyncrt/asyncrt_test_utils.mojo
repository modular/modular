# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.param_env import env_get_string, is_defined

from gpu.host import DeviceContext


fn expect_eq[
    type: DType, size: Int
](val: SIMD[type, size], expected: SIMD[type, size], msg: String = "") raises:
    if val != expected:
        raise Error("expect_eq failed: " + msg)


fn expect_eq(val: Bool, expected: Bool, msg: String = "") raises:
    if val != expected:
        raise Error("expect_eq failed: " + msg)


fn is_v2_context() -> Bool:
    return is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()


fn device_kind() -> StringLiteral:
    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        return env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()
    else:
        return "default"


fn create_test_device_context(gpu_id: Int = 0) raises -> DeviceContext:
    # Create an instance of the DeviceContext
    var test_ctx: DeviceContext

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        var kind = env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()
        print("Using DeviceContext: V2 - " + str(kind))
        test_ctx = DeviceContext(kind, gpu_id=gpu_id)
    else:
        print("Using DeviceContext: default")
        test_ctx = DeviceContext(gpu_id=gpu_id)

    return test_ctx
