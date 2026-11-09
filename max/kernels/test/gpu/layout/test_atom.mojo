# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for `layout.atoms`: each atom is issued directly on one tile.

Every tile is an `OwnedGPUTensor` in a kernel: shared memory for the memory
operands, registers for the register ones. The kernels run on one thread.
"""

from layout import (
    ImmTileTensor,
    MutTileTensor,
    TileTensor,
    coord,
    row_major,
)
from layout.atoms import (
    INDEX_LANE,
    INDEX_VALUE,
    CastAtom,
    AddressEncoding,
    MmaAtom,
    OperandMap,
    Synchronous,
    registers_map,
    vector_map,
)
from layout.bit_layout import BitLayout, bits
from layout.owned_tensor import OwnedGPUTensor
from layout.atoms.pointer import PointerLoad, PointerStore
from layout.layout import TensorLayout
from layout.tensor_engine import DefaultEngine, TensorEngine

from max.gpu.host import DeviceContext

from std.sys import size_of

from std.testing import assert_equal, assert_false, assert_true, TestSuite


comptime Regs = DefaultEngine[element_width=1]
comptime Out = MutPointer[Float32, _]

comptime N = 8


@fieldwise_init
struct Cvt[src_dt: DType, dst_dt: DType, width: Int](CastAtom):
    """`width` adjacent elements in one conversion: `cvt.rn.bf16x2.f32` or
    `v_cvt_pk_bf16_f32` at width 2, a scalar conversion at width 1."""

    comptime shape_values = ParameterList.of[Self.width]().values
    comptime Completion = Synchronous
    comptime src_dtype = Self.src_dt
    comptime dst_dtype = Self.dst_dt
    comptime SrcEngine = Regs
    comptime DstEngine = Regs
    comptime src_map = vector_map(Self.width, size_of[Self.src_dt]())
    comptime dst_map = vector_map(Self.width, size_of[Self.dst_dt]())

    def __call__[
        TD: DType,
        LD: TensorLayout,
        ED: TensorEngine,
        TS: DType,
        LS: TensorLayout,
        ES: TensorEngine,
    ](
        self,
        dst: MutTileTensor[TD, LD, Engine=ED, ...],
        src: ImmTileTensor[TS, LS, Engine=ES, ...],
    ):
        dst.store(
            coord[0, 0], src.load[width=Self.width](coord[0, 0]).cast[TD]()
        )


def _broadcast(values: Int, element_bytes: Int) -> OperandMap:
    """Every value reads the one element at the lane's address."""
    var fixed = BitLayout()
    var n = 0
    while (1 << n) < values:
        fixed = fixed.put(INDEX_VALUE + n, 0)
        n += 1
    return {
        AddressEncoding.BYTES,
        fixed,
        UInt64(element_bytes - 1),
    }


@fieldwise_init
struct Fma[dt: DType, width: Int](MmaAtom):
    """A 1 x `width` x 1 multiply-accumulate: one A element times `width`
    adjacent B elements, added into `width` adjacent accumulators."""

    comptime shape_values = ParameterList.of[1, Self.width, 1]().values
    comptime Completion = Synchronous
    comptime a_dtype = Self.dt
    comptime b_dtype = Self.dt
    comptime c_dtype = Self.dt
    comptime AEngine = Regs
    comptime BEngine = Regs
    comptime CEngine = Regs
    # Every value multiplies the one A element.
    comptime a_map = _broadcast(Self.width, size_of[Self.dt]())
    comptime b_map = vector_map(Self.width, size_of[Self.dt]())
    comptime c_map = vector_map(Self.width, size_of[Self.dt]())

    def __call__[
        TC: DType,
        LC: TensorLayout,
        EC: TensorEngine,
        TA: DType,
        LA: TensorLayout,
        EA: TensorEngine,
        TB: DType,
        LB: TensorLayout,
        EB: TensorEngine,
    ](
        self,
        c: MutTileTensor[TC, LC, Engine=EC, ...],
        a: ImmTileTensor[TA, LA, Engine=EA, ...],
        b: ImmTileTensor[TB, LB, Engine=EB, ...],
    ):
        var x = a.load[width=1](coord[0, 0]).cast[TC]()
        var y = b.load[width=Self.width](coord[0, 0]).cast[TC]()
        c.store(
            coord[0, 0],
            c.load[width=Self.width](coord[0, 0])
            + SIMD[TC, Self.width](x[0]) * y,
        )


def run[
    kernel: def[o: MutOrigin](Out[o]) thin -> None
](ctx: DeviceContext) raises -> List[Float32]:
    var buf = ctx.enqueue_create_buffer[.float32](N)
    buf.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel[origin_of(buf)]](
        buf.unsafe_ptr(),
        grid_dim=1,
        block_dim=1,
    )
    var host = ctx.enqueue_create_host_buffer[.float32](N)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var result = List[Float32](capacity=N)
    for i in range(N):
        result.append(host[i])
    return result^


def load_store_kernel(output: Out):
    var a = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    var r = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    for i in range(4):
        TileTensor(a)[0, i] = Float32(i + 1)
    _ = TileTensor(r).fill(0.0)

    var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 4]())
    PointerLoad[DType.float32, 4]()(TileTensor(regs), TileTensor(a))
    PointerStore[DType.float32, 4]()(TileTensor(r), TileTensor(regs))
    for i in range(4):
        output.unsafe_store(i, TileTensor(r)[0, i][0])


def test_load_and_store_atoms() raises:
    with DeviceContext() as ctx:
        var r = run[load_store_kernel](ctx)
        for i in range(4):
            assert_equal(r[i], Float32(i + 1))


def cast_kernel(output: Out):
    var a = OwnedGPUTensor[.float32, .SHARED](row_major[1, 2]())
    var h = OwnedGPUTensor[.bfloat16, .SHARED](row_major[1, 2]())
    TileTensor(a)[0, 0] = 3.0
    TileTensor(a)[0, 1] = 5.0

    Cvt[DType.float32, DType.bfloat16, 2]()(TileTensor(h), TileTensor(a))
    for i in range(2):
        output.unsafe_store(i, TileTensor(h)[0, i][0].cast[.float32]())


def test_cast_atom() raises:
    with DeviceContext() as ctx:
        var r = run[cast_kernel](ctx)
        assert_equal(r[0], Float32(3))
        assert_equal(r[1], Float32(5))


def mma_kernel(output: Out):
    # A is padded to 16 bytes so B and C stay aligned for 4-wide loads.
    var a = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    var b = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    var c = OwnedGPUTensor[.float32, .SHARED](row_major[1, 4]())
    TileTensor(a)[0, 0] = 2.0
    for j in range(4):
        TileTensor(b)[0, j] = Float32(j)
        TileTensor(c)[0, j] = Float32(10 * j)

    Fma[DType.float32, 4]()(TileTensor(c), TileTensor(a), TileTensor(b))
    for j in range(4):
        output.unsafe_store(j, TileTensor(c)[0, j][0])


def test_mma_atom_accumulates() raises:
    with DeviceContext() as ctx:
        var r = run[mma_kernel](ctx)
        for j in range(4):
            assert_equal(r[j], Float32(12 * j))


def test_shapes() raises:
    comptime load = PointerLoad[DType.float32, 4].shape
    assert_equal(load.length, 1)
    assert_equal(load[0], 4)
    comptime fma = Fma[DType.float32, 4].shape
    assert_equal(fma.length, 3)
    assert_equal(fma[0], 1)
    assert_equal(fma[1], 4)
    assert_equal(fma[2], 1)


def test_maps() raises:
    comptime load = PointerLoad[DType.float32, 4].mem_map
    # Value v is 4 v bytes past the lane's address; the lane is free but
    # must keep clear of the 16 bytes one access covers.
    assert_equal(load.fixed(UInt64(3)), UInt64(12))
    assert_equal(load.free(bits(INDEX_LANE, 5) | 3), bits(INDEX_LANE, 5))
    assert_equal(load.avoid, UInt64(15))
    comptime pair = Cvt[DType.float32, DType.bfloat16, 2]
    assert_equal(pair.dst_map.fixed(UInt64(1)), UInt64(2))
    comptime fma = Fma[DType.float32, 4]
    # A broadcasts: every value reads the same element, while C steps.
    assert_equal(fma.a_map.fixed(UInt64(3)), UInt64(0))
    assert_equal(fma.a_map.fixed.live(), UInt64(3))
    assert_equal(fma.c_map.fixed(UInt64(2)), UInt64(8))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
