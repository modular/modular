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

from std.os import abort
from std.random import rand

from std.benchmark import *
from std.benchmark import keep
from layout import Coord, RuntimeInt, TileTensor, row_major
from linalg.matmul import matmul
from linalg.packing import pack_b_ndbuffer, pack_matmul_b_shape_func
from std.testing import assert_almost_equal


def _ri(v: Int) -> RuntimeInt[DType.int64]:
    return RuntimeInt[DType.int64](Int64(v))


def gemm_naive(a: TileTensor, b: TileTensor, c: TileTensor[mut=True, ...]):
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())
    _ = c.fill(Scalar[c.dtype](0))

    var ak = Int(a.dim[1]())
    var bn = Int(b.dim[1]())

    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a.ptr[i * ak + p].cast[c.dtype]()
                var b_val = b.ptr[p * bn + j].cast[c.dtype]()
                c.ptr[i * n + j] += a_val * b_val


def verify(a: TileTensor, b: TileTensor, c: TileTensor):
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())

    var c_ref_ptr = alloc[Scalar[c.dtype]](m * n)
    var c_ref = TileTensor(c_ref_ptr, row_major(Coord(_ri(m), _ri(n))))
    gemm_naive(a, b, c_ref)

    for i in range(m):
        for j in range(n):
            try:
                assert_almost_equal(c.ptr[i * n + j], c_ref.ptr[i * n + j])
            except e:
                abort(String(e))
    c_ref_ptr.free()


def bench_matmul_spec(mut m: Bench, spec: MatmulSpec) raises:
    # disatch to bench_matmul with concrete spec type
    m.bench_with_input[
        MatmulSpec[spec.static_info], bench_matmul[spec.static_info]
    ](
        BenchId("matmul", String(spec)),
        spec,
        # TODO: Pick relevant benchmetric
        [ThroughputMeasure(BenchMetric.elements, spec.flops())],
    )


def bench_matmul[
    static: MatmulSpecStatic
](mut bencher: Bencher, spec: MatmulSpec[static]) raises capturing:
    comptime a_type = spec.static_info.a_type
    comptime b_type = spec.static_info.b_type
    comptime c_type = spec.static_info.c_type
    comptime b_packed = spec.static_info.b_packed
    comptime alignment = 64
    var a_ptr = alloc[Scalar[a_type],](spec.m * spec.k, alignment=alignment)
    var b_ptr = alloc[Scalar[b_type],](spec.k * spec.n, alignment=alignment)
    var c_ptr = alloc[Scalar[c_type],](spec.m * spec.n, alignment=alignment)
    var a = TileTensor(a_ptr, row_major(Coord(_ri(spec.m), _ri(spec.k))))
    var b = TileTensor(b_ptr, row_major(Coord(_ri(spec.k), _ri(spec.n))))
    var c = TileTensor(c_ptr, row_major(Coord(_ri(spec.m), _ri(spec.n))))
    rand[a_type](a_ptr, spec.m * spec.k)
    rand[b_type](b_ptr, spec.k * spec.n)
    _ = c.fill(Scalar[c_type](0))

    var padded_n_k = pack_matmul_b_shape_func[a_type, c_type, False](b)

    var padded_n = padded_n_k[1] if b_packed else spec.n
    var padded_k = padded_n_k[0] if b_packed else spec.k

    var bp_ptr = alloc[Scalar[b_type],](
        padded_k * padded_n, alignment=alignment
    )
    var bp = TileTensor(bp_ptr, row_major(Coord(_ri(padded_k), _ri(padded_n))))

    if b_packed:
        pack_b_ndbuffer[a_type, c_type](b, bp)

    @always_inline
    @parameter
    def bench_fn() raises:
        matmul[
            transpose_b=False,
            b_packed=b_packed,
            saturated_vnni=False,
        ](c, a, bp if b_packed else b)
        keep(c.ptr)

    bencher.iter[bench_fn]()
    verify(a, b, c)

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c_ptr.free()


@fieldwise_init
struct MatmulSpecStatic(ImplicitlyCopyable):
    var b_packed: Bool
    var a_type: DType
    var b_type: DType
    var c_type: DType


@fieldwise_init
struct MatmulSpec[static_info: MatmulSpecStatic](ImplicitlyCopyable, Writable):
    var m: Int
    var n: Int
    var k: Int

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the matmul spec.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "m=",
            self.m,
            ";n=",
            self.n,
            ";k=",
            self.k,
            ";b_packed=",
            Self.static_info.b_packed,
            ";a_type=",
            Self.static_info.a_type,
            ";b_type=",
            Self.static_info.b_type,
            ";c_type=",
            Self.static_info.c_type,
        )

    def flops(self) -> Int:
        return 2 * self.m * self.n * self.k


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))

    comptime packed_float32 = MatmulSpecStatic(
        b_packed=True,
        a_type=DType.float32,
        b_type=DType.float32,
        c_type=DType.float32,
    )
    comptime unpacked_float32 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.float32,
        b_type=DType.float32,
        c_type=DType.float32,
    )

    bench_matmul_spec(m, MatmulSpec[packed_float32](m=256, n=256, k=256))
    bench_matmul_spec(m, MatmulSpec[packed_float32](m=512, n=512, k=512))
    bench_matmul_spec(m, MatmulSpec[packed_float32](m=1024, n=1024, k=1024))
    bench_matmul_spec(m, MatmulSpec[unpacked_float32](m=256, n=256, k=256))

    m.dump_report()
