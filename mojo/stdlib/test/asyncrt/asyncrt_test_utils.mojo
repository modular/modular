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


fn kind() -> String:
    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        alias kind = env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()

        @parameter
        if kind == "gpu":
            return DeviceContext.device_kind
        return kind
    return "default"


fn create_test_device_context(device_id: Int = 0) raises -> DeviceContext:
    # Create an instance of the DeviceContext
    var test_ctx: DeviceContext

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        print("Using DeviceContext: V2 - " + kind())
        test_ctx = DeviceContext(device_id=device_id, kind=kind())
    elif is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V1"]():
        raise Error("DeviceContextV1 is unsupported")
    else:
        print("Using DeviceContext: default")
        test_ctx = DeviceContext(device_id=device_id)

    return test_ctx
