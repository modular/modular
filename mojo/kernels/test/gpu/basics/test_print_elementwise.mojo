# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from os import abort
from sys import simdwidthof

from algorithm.functional import elementwise
from gpu import block_idx, thread_idx
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from layout import Layout, LayoutTensor, RuntimeLayout
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import UNKNOWN_VALUE, IntTuple

from utils.index import IndexList


fn test_elementwise_print[
    c_type: DType,
    c_layout: Layout,
](
    c01: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    var M = c01.dim(0)
    var N = c01.dim(1) // 2
    alias simd_width = simdwidthof[
        c_type, target = _get_gpu_target["sm_80"]()
    ]()

    @always_inline
    @__copy_capture(c01, N)
    @parameter
    fn binary[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var m: Int = idx0[0]
        var n: Int = idx0[1]
        print("print thousands of messages: m=", m, " n=", n, sep="")

    print("about to call elementwise, M=", M, "N=", N)
    elementwise[binary, simd_width, target="gpu"](IndexList[2](M, N), ctx)
    print("called elementwise")


fn runtime_row_major[
    cols: Int
](
    rows: Int,
    out res: RuntimeLayout[
        Layout(IntTuple(UNKNOWN_VALUE, cols), IntTuple(cols, 1))
    ],
):
    return __type_of(res).row_major(IndexList[2]((rows, cols)))


fn test_dual_matmul[
    N: Int = 512, K: Int = 512
](ctx: DeviceContext, M: Int = 512) raises:
    alias dst_type = DType.float32
    var layout_c01 = runtime_row_major[2 * N](M)
    var mat_c01 = ManagedLayoutTensor[dst_type](layout_c01, ctx)
    test_elementwise_print(
        mat_c01.device_tensor(),
        ctx,
    )
    # CHECK: returned from test_elementwise_print
    print("returned from test_elementwise_print")
    _ = mat_c01^


fn main() raises:
    with DeviceContext() as ctx:
        test_dual_matmul(ctx)
