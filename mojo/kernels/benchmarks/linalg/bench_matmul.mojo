# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Matmul import matmul, pack_b_ndbuffer, pack_matmul_b_shape_func
from memory.buffer import NDBuffer
from utils.index import Index

from mojobench import *
from benchmark import keep


fn bench_matmul(inout m: MojoBench, spec: MatmulSpec) raises:
    @parameter
    @always_inline
    fn bench_matmul_wrapper(
        inout b: Bencher, concrete_spec: MatmulSpec[spec.static_info]
    ):
        bench_matmul(b, concrete_spec)

    m.bench_with_input[MatmulSpec[spec.static_info], bench_matmul_wrapper](
        BenchId("matmul", str(spec)),
        spec,
        throughput_elems=spec.flops(),
    )


fn bench_matmul(inout bencher: Bencher, spec: MatmulSpec) capturing:
    alias a_type = spec.static_info.a_type
    alias b_type = spec.static_info.b_type
    alias c_type = spec.static_info.c_type
    alias b_packed = spec.static_info.b_packed
    alias alignment = 64
    let a_ptr = DTypePointer[a_type].aligned_alloc(alignment, spec.m * spec.k)
    let b_ptr = DTypePointer[b_type].aligned_alloc(alignment, spec.k * spec.n)
    let b = NDBuffer[b_type, 2](b_ptr, Index(spec.k, spec.n))
    b.zero()

    var padded_n_k = StaticIntTuple[2]()
    padded_n_k = pack_matmul_b_shape_func[
        a_type,
        DimList.create_unknown[2](),
        b_type,
        DimList.create_unknown[2](),
        c_type,
        DimList.create_unknown[2](),
        transpose_in_0=False,
        single_thread_blocking_override=False,
    ](b)

    let padded_n = padded_n_k[1] if b_packed else spec.n
    let padded_k = padded_n_k[0] if b_packed else spec.k

    let bp_ptr = DTypePointer[b_type].aligned_alloc(
        alignment, padded_k * padded_n
    )
    let c0_ptr = DTypePointer[c_type].aligned_alloc(alignment, spec.m * spec.n)
    let c1_ptr = DTypePointer[c_type].aligned_alloc(alignment, spec.m * spec.n)

    var a = NDBuffer[a_type, 2](a_ptr, Index(spec.m, spec.k))
    a.zero()

    var bp = NDBuffer[b_type, 2](bp_ptr, Index(padded_k, padded_n))
    var c = NDBuffer[c_type, 2](c0_ptr, Index(spec.m, spec.n))

    if b_packed:
        pack_b_ndbuffer[
            a_type,
            DimList.create_unknown[2](),
            b_type,
            DimList.create_unknown[2](),
            c_type,
            DimList.create_unknown[2](),
        ](b, bp)

    @always_inline
    @parameter
    fn bench_fn():
        matmul[
            transpose_b=False,
            b_packed=b_packed,
            saturated_vnni=False,
        ](c, a, bp)
        keep(c.data)

    bencher.iter[bench_fn]()

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()
    c1_ptr.free()


@value
struct MatmulSpecStatic:
    var b_packed: Bool
    var a_type: DType
    var b_type: DType
    var c_type: DType


@value
struct MatmulSpec[static_info: MatmulSpecStatic](Stringable):
    var m: Int
    var n: Int
    var k: Int

    fn __str__(self) -> String:
        return (
            "m="
            + str(self.m)
            + ";n="
            + str(self.n)
            + ";k="
            + str(self.k)
            + ";b_packed="
            + str(Self.static_info.b_packed)
            + ";a_type="
            + str(Self.static_info.a_type)
            + ";b_type="
            + str(Self.static_info.b_type)
            + ";c_type="
            + str(Self.static_info.c_type)
        )

    fn flops(self) -> Int:
        return 2 * self.m * self.n * self.k


def main():
    var m = MojoBench(MojoBenchConfig(num_repetitions=2))

    alias packed_float32 = MatmulSpecStatic(
        b_packed=True,
        a_type=DType.float32,
        b_type=DType.float32,
        c_type=DType.float32,
    )
    alias unpacked_float32 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.float32,
        b_type=DType.float32,
        c_type=DType.float32,
    )

    bench_matmul(m, MatmulSpec[packed_float32](m=256, n=256, k=256))
    bench_matmul(m, MatmulSpec[packed_float32](m=512, n=512, k=512))
    bench_matmul(m, MatmulSpec[packed_float32](m=1024, n=1024, k=1024))
    bench_matmul(m, MatmulSpec[unpacked_float32](m=256, n=256, k=256))

    m.dump_report()
