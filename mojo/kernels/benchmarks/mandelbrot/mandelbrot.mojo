# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This uses mandelbrot as an example to test how the entire stdlib works
# together.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s -execute | FileCheck %s


from DType import DType
from SIMD import F32
from SIMD import SIMD
from IO import print, _printf
from Range import range
from Vector import DynamicVector
from Buffer import Buffer, NDBuffer
from Matrix import Matrix
from List import Dim, DimList
from LLCL import num_cores, Runtime, TaskGroup
from Functional import parallelize
from Math import iota
from Complex import ComplexSIMD
from Benchmark import Benchmark
from Assert import assert_param_msg
from Pointer import Pointer, DTypePointer

alias float_type = DType.f64
alias int_type = DType.si64


fn draw_mandelbrot[
    h: __mlir_type.index, w: __mlir_type.index
](out: Matrix[DimList(h, w), int_type, False]):
    let sr = StringRef(".,c8M@jawrpogOQEPGJ")
    let charset = Buffer[Dim(), DType.si8](sr.data.address, sr.length)
    for row in range(h):
        for col in range(w):
            let v: Int = out[row, col].value
            if v > 0:
                let p = charset[v % sr.length]
                _printf[DType.si8]("%c", p.value)
            else:
                print("0")
        print("\n")


fn mandelbrot_kernel[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width], iter: Int) -> SIMD[
    int_type, simd_width
]:
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var nv: SIMD[int_type, simd_width] = 0
    var mask: SIMD[DType.bool, simd_width] = -1
    var i: Int = iter

    while (i != 0) and mask.reduce_or():
        mask = z.norm() <= 4
        z = z.sq_add(c)
        nv = mask.select(nv + 1, nv)
        i -= 1
    return nv


fn mandelbrot[
    simd_width: Int,
    h: __mlir_type.index,
    w: __mlir_type.index,
    iter: Int,
    parallel: Bool,
](
    out: Matrix[DimList(h, w), int_type, False],
    rows_per_worker: Int,
    min_x: SIMD[float_type, simd_width],
    max_x: SIMD[float_type, simd_width],
    min_y: SIMD[float_type, simd_width],
    max_y: SIMD[float_type, simd_width],
    rt: Runtime,
):
    # Each task gets a row
    @always_inline
    fn worker(row: Int):
        let rowv: SIMD[float_type, simd_width] = row
        let simd_val = iota[simd_width, float_type]()
        let scalex = (max_x - min_x) / w
        let scaley = (max_y - min_y) / h
        for col in range(w // simd_width):
            var colv: SIMD[float_type, simd_width] = col * simd_width
            colv = colv + simd_val
            let cx = min_x + colv * scalex
            let cy = min_y + rowv * scaley
            let c = ComplexSIMD[simd_width, float_type](cx, cy)
            out.simd_store[simd_width](
                row, col * simd_width, mandelbrot_kernel[simd_width](c, iter)
            )

    @parameter
    if parallel:
        parallelize[worker](rt, h)
    else:
        for row in range(h):
            worker(row)


@export
fn main():
    let rt = Runtime()

    alias width = 4096
    # using simd_width=16
    assert_param_msg[width % 16 == 0, "must be a multiple of 16"]()
    alias height = 4096
    alias iter = 1000

    let vec = DynamicVector[__mlir_type[`!pop.scalar<`, int_type.value, `>`]](
        width * height
    )
    let dptr = DTypePointer[int_type](vec.data.address)
    let m: Matrix[DimList(height, width), int_type, False] = vec.data

    @always_inline
    fn bench_parallel[simd_width: __mlir_type.index]():
        let min_x = -2.0
        let max_x = 0.47
        let min_y = -1.12
        let max_y = 1.12

        mandelbrot[simd_width, height, width, iter, True](
            m, 1, min_x, max_x, min_y, max_y, rt
        )

    var time: F32
    let ns_per_second: Int = 1_000_000_000

    bench_parallel[16]()
    var pixel_sum: Int = 0
    for i in range(height):
        for j in range(width):
            pixel_sum += m[i, j].value
    print("pixel sum: ")
    print(pixel_sum)

    var num_warmup: Int = 1
    time = Benchmark(num_warmup).run[bench_parallel[16]]()
    time = time / ns_per_second
    print(time)

    time = Benchmark(num_warmup).run[bench_parallel[8]]()
    time = time / ns_per_second
    print(time)

    time = Benchmark(num_warmup).run[bench_parallel[4]]()
    time = time / ns_per_second
    print(time)

    time = Benchmark(num_warmup).run[bench_parallel[1]]()
    time = time / ns_per_second
    print(time)

    print(m[0, 0])
    # draw_mandelbrot[height, width](m)
    vec.del_old()
