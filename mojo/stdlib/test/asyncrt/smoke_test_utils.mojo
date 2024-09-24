# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.host import DeviceContextVariant
from sys.param_env import is_defined, env_get_string


fn expect_eq(val: Int, expected: Int, msg: String = "") raises:
    if val != expected:
        raise Error("expect_eq failed: " + msg)


fn expect_eq(val: Float32, expected: Float32, msg: String = "") raises:
    if val != expected:
        raise Error("expect_eq failed: " + msg)


fn expect_eq[T: EqualityComparable](val: T, expected: T, msg: String = ""):
    if val != expected:
        print("#$# failed equality check: " + msg)


fn create_test_device_context() raises -> DeviceContextVariant:
    # Create an instance of the DeviceContextVariant
    var test_ctx: DeviceContextVariant

    print(
        "Using DeviceContextVariant for "
        + env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2", "V1"]()
    )

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        test_ctx = DeviceContextVariant(
            env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()
        )
    else:
        test_ctx = DeviceContextVariant()

    return test_ctx
