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


fn create_test_device_context() raises -> DeviceContext:
    # Create an instance of the DeviceContext
    var test_ctx: DeviceContext

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        var kind = env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()
        print("Using DeviceContext: V2 - " + kind)
        test_ctx = DeviceContext(kind)
    else:
        print("Using DeviceContext: default")
        test_ctx = DeviceContext()

    return test_ctx
