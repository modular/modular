# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import NDBuffer
from gpu import ThreadIdx
from gpu.host import DeviceContext
from internal_utils import HostNDBuffer

from utils.index import IndexList


@register_passable("trivial")
trait BaseT:
    fn get_val(self, idx: Int) -> Float32:
        ...


@value
@register_passable("trivial")
struct ImplT(BaseT):
    alias rank = 1
    var values: NDBuffer[DType.float32, Self.rank]

    def __init__(out self, buf: __type_of(self.values)):
        self.values = buf

    fn get_val(self, idx: Int) -> Float32:
        return self.values[idx]


def trait_repro_sub[
    t: BaseT
](thing: t, ctx: DeviceContext, size: Int,):
    @parameter
    @__copy_capture(thing)
    fn kernel_fn():
        var idx = ThreadIdx.x
        print(Float32(thing.get_val(idx)) * 2)

    cuda_fn = ctx.compile_function[kernel_fn]()
    ctx.enqueue_function(cuda_fn, grid_dim=(1,), block_dim=(size))


def trait_repro(ctx: DeviceContext):
    var size = 5
    var host_buf = HostNDBuffer[DType.float32, 1]((size,))
    for i in range(size):
        host_buf.tensor[i] = i

    var device_buf = host_buf.copy_to_device(ctx)
    var device_nd = device_buf.tensor
    var thing = ImplT(device_nd)
    trait_repro_sub(thing, ctx, size)
    ctx.enqueue_copy_from_device(host_buf.tensor.data, device_buf.buffer)
    ctx.synchronize()

    for i in range(size):
        print(host_buf.tensor[i])
    print("yay!")

    _ = device_buf^


# CHECK: yay!
def main():
    with DeviceContext() as ctx:
        trait_repro(ctx)
