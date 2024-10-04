# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.host import DeviceContextVariant, DeviceBufferVariant
from smoke_test_utils import expect_eq


fn _ownership_helper(
    owned ctx: DeviceContextVariant,
) raises -> DeviceContextVariant:
    var ctx_copy = ctx
    print("local ctx_copy: " + ctx_copy.name())
    return ctx_copy


fn _ownership_helper_buf[
    type: DType
](owned buf: DeviceBufferVariant[type]) raises -> DeviceBufferVariant[type]:
    var buf_copy = buf
    print("local buf_copy: " + str(len(buf)))
    return buf_copy


fn _run_ownership_transfer(ctx: DeviceContextVariant) raises:
    print("-")
    print("run_ownership_transfer()")

    var ctx_copy = _ownership_helper(ctx)
    print("ctx_copy: " + ctx_copy.name())

    var buf = ctx.create_buffer_sync[DType.float32](32)
    print("buf: " + str(len(buf)))
    var buf_copy = _ownership_helper_buf(buf)
    print("buf_copy: " + str(len(buf_copy)))

    _ = buf
    _ = ctx


fn test_smoke(ctx: DeviceContextVariant) raises:
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    _run_ownership_transfer(ctx)

    print("Done.")
