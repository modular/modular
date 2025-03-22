# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Note: CPU function compilation not supported
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext, DeviceMulticastBuffer


fn test_multicast_memory(contexts: List[DeviceContext]) raises:
    alias alloc_len = 128 * 1024
    alias dtype = DType.int32

    var multicast_buf = DeviceMulticastBuffer[dtype](contexts, alloc_len)

    for i in range(len(contexts)):
        var dev_buf = multicast_buf.unicast_buffer_for(contexts[i])
        with dev_buf.map_to_host() as host_buf:
            for i in range(len(host_buf)):
                host_buf[i] = i * 2

    print(multicast_buf.unicast_buffer_for(contexts[0]))


fn main() raises:
    var ctx0 = create_test_device_context(device_id=0)
    if not ctx0.supports_multicast():
        print("Multicast memory not supported")
        return

    var num_gpus = DeviceContext.number_of_devices(api="gpu")
    if num_gpus < 2:
        print("Too few devices")
        return

    var ctx1 = create_test_device_context(device_id=1)

    test_multicast_memory(List(ctx0, ctx1))
