# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.param_env import env_get_string, is_defined

from gpu.host import DeviceContextVariant


fn expect_eq[
    type: DType, size: Int
](val: SIMD[type, size], expected: SIMD[type, size], msg: String = "") raises:
    if val != expected:
        raise Error("expect_eq failed: " + msg)


fn create_test_device_context() raises -> DeviceContextVariant:
    # Create an instance of the DeviceContextVariant
    var test_ctx: DeviceContextVariant

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        var kind = env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()
        print("Using DeviceContextVariant: V2 - " + kind)
        test_ctx = DeviceContextVariant(kind)
    elif is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V1"]():
        var kind = env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V1"]()
        print("Using DeviceContextVariant: V1 - " + kind)
        test_ctx = DeviceContextVariant(kind)
    else:
        print("Using DeviceContextVariant: default")
        test_ctx = DeviceContextVariant()

    return test_ctx
