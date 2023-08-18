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
# RUN: %mojo %s | FileCheck %s


from builtin.io import _printf
from Vector import DynamicVector
from Buffer import Buffer, NDBuffer
from Matrix import Matrix
from List import Dim, DimList
from runtime.llcl import num_cores, Runtime, TaskGroup
from Functional import parallelize
from math import iota
from complex import ComplexSIMD
from Benchmark import Benchmark
from Assert import assert_param
from Pointer import Pointer, DTypePointer
from sys.info import simdwidthof
from TypeUtilities import rebind
from math import abs

alias float_type = DType.float64
alias int_type = DType.index


alias width = 4096
# using simd_width=16
assert_param[width % 16 == 0, "must be a multiple of 16"]()
alias height = 4096
alias MAX_ITERS = 1000


fn draw_mandelbrot[h: Int, w: Int](out: Matrix[DimList(h, w), int_type, False]):
    let sr = StringRef(".,c8M@jawrpogOQEPGJ")
    let charset = Buffer[Dim(), DType.int8](
        rebind[DTypePointer[DType.int8]](sr.data), sr.length
    )
    for row in range(h):
        for col in range(w):
            let v: Int = out[row, col].value
            if v > 0:
                let p = charset[v % sr.length]
                _printf("%c", p.value)
            else:
                print_no_newline("0")
        print("")


# ===----------------------------------------------------------------------===#
# Blog post 1
# ===----------------------------------------------------------------------===#


@always_inline
def mandelbrot_kernel_part0(c: ComplexSIMD[float_type, 1]) -> Int:
    z = ComplexSIMD[float_type, 1](0, 0)
    nv = 0

    for i in range(1, MAX_ITERS):
        if abs(z) > 2:
            break
        z = z * z + c
        nv += 1
    return nv


@always_inline
fn mandelbrot_kernel_part1(c: ComplexSIMD[float_type, 1]) -> Int:
    var z = ComplexSIMD[float_type, 1](0, 0)
    var nv = 0

    for i in range(1, MAX_ITERS):
        if abs(z) > 2:
            break
        z = z * z + c
        nv += 1
    return nv


@always_inline
fn mandelbrot_kernel_part2(c: ComplexSIMD[float_type, 1]) -> Int:
    var z = ComplexSIMD[float_type, 1](0, 0)
    var nv = 0

    for i in range(MAX_ITERS):
        if z.squared_norm() > 4:
            break
        z = z.squared_add(c)
        nv += 1
    return nv


@always_inline
fn mandelbrot_blog_1[
    h: Int, w: Int, part: Int
](
    out: Matrix[DimList(h, w), int_type, False],
    min_x: SIMD[float_type, 1],
    max_x: SIMD[float_type, 1],
    min_y: SIMD[float_type, 1],
    max_y: SIMD[float_type, 1],
):
    let scalex = (max_x - min_x) / w
    let scaley = (max_y - min_y) / h

    for row in range(h):
        for col in range(w):
            let cx = min_x + col * scalex
            let cy = min_y + row * scaley
            let c = ComplexSIMD[float_type, 1](cx, cy)

            let res: SIMD[int_type, 1]

            @parameter
            if part == 0:
                try:
                    res = mandelbrot_kernel_part0(c)
                except:
                    res = 0
            elif part == 1:
                res = mandelbrot_kernel_part1(c)
            elif part == 2:
                res = mandelbrot_kernel_part2(c)
            else:
                res = 0

            out[row, col] = res


# ===----------------------------------------------------------------------===#
# Optimized
# ===----------------------------------------------------------------------===#


@always_inline
fn mandelbrot_kernel[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var nv: SIMD[int_type, simd_width] = 0
    var mask: SIMD[DType.bool, simd_width] = -1
    var i: Int = MAX_ITERS

    while (i != 0) and mask.reduce_or():
        mask = z.squared_norm() <= 4
        z = z.squared_add(c)
        nv = mask.select(nv + 1, nv)
        i -= 1
    return nv


@always_inline
fn mandelbrot[
    simd_width: Int, h: Int, w: Int, parallel: Bool
](
    out: Matrix[DimList(h, w), int_type, False],
    rows_per_worker: Int,
    min_x: SIMD[float_type, simd_width],
    max_x: SIMD[float_type, simd_width],
    min_y: SIMD[float_type, simd_width],
    max_y: SIMD[float_type, simd_width],
):
    # Each task gets a row
    @always_inline
    @parameter
    fn worker(row: Int):
        let rowv: SIMD[float_type, simd_width] = row
        let simd_val = iota[float_type, simd_width]()
        let scalex = (max_x - min_x) / w
        let scaley = (max_y - min_y) / h
        for col in range(w // simd_width):
            var colv: SIMD[float_type, simd_width] = col * simd_width
            colv = colv + simd_val
            let cx = min_x + colv * scalex
            let cy = min_y + rowv * scaley
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            out.simd_store[simd_width](
                row, col * simd_width, mandelbrot_kernel[simd_width](c)
            )

    @parameter
    if parallel:
        parallelize[worker](h)
    else:
        for row in range(h):
            worker(row)


fn main():
    let vec = DynamicVector[__mlir_type[`!pop.scalar<`, int_type.value, `>`]](
        width * height
    )
    let dptr = DTypePointer[int_type](vec.data.address)
    let m: Matrix[DimList(height, width), int_type, False] = vec.data

    @always_inline
    @parameter
    fn bench_fn[part: Int]():
        let min_x = -2.0
        let max_x = 0.47
        let min_y = -1.12
        let max_y = 1.12

        mandelbrot_blog_1[height, width, part](m, min_x, max_x, min_y, max_y)

    var time: Float64
    let ns_per_second: Int = 1_000_000_000

    bench_fn[2]()
    var pixel_sum: Int = 0
    for i in range(height):
        for j in range(width):
            pixel_sum += m[i, j].to_int()
    print("pixel sum: ", pixel_sum)

    var num_warmup: Int = 1
    time = Benchmark(num_warmup).run[bench_fn[0]]() / ns_per_second
    print("blog post 1 with part=0 ", time)

    time = Benchmark(num_warmup).run[bench_fn[1]]() / ns_per_second
    print("blog post 1 with part=1 ", time)

    time = Benchmark(num_warmup).run[bench_fn[2]]() / ns_per_second
    print("blog post 1 with part=2 ", time)

    vec._del_old()


fn main_efficient():

    let vec = DynamicVector[__mlir_type[`!pop.scalar<`, int_type.value, `>`]](
        width * height
    )
    let dptr = DTypePointer[int_type](vec.data.address)
    let m: Matrix[DimList(height, width), int_type, False] = vec.data

    @always_inline
    @parameter
    fn bench_parallel[simd_width: Int]():
        let min_x = -2.0
        let max_x = 0.47
        let min_y = -1.12
        let max_y = 1.12

        mandelbrot[simd_width, height, width, True](
            m, 1, min_x, max_x, min_y, max_y
        )

    var time: Float64
    let ns_per_second: Int = 1_000_000_000

    bench_parallel[simdwidthof[DType.float32]()]()
    var pixel_sum: Int = 0
    for i in range(height):
        for j in range(width):
            pixel_sum += m[i, j].to_int()
    print("pixel sum: ", pixel_sum)

    var num_warmup: Int = 1
    time = Benchmark(num_warmup).run[bench_parallel[16]]() / ns_per_second
    print("Parallel with simd_width=16 ", time)

    time = Benchmark(num_warmup).run[bench_parallel[8]]() / ns_per_second
    print("Parallel with simd_width=8 ", time)

    time = Benchmark(num_warmup).run[bench_parallel[4]]() / ns_per_second
    print("Parallel with simd_width=4 ", time)

    time = Benchmark(num_warmup).run[bench_parallel[1]]() / ns_per_second
    print("Parallel with simd_width=1 ", time)

    print(m[0, 0])
    # draw_mandelbrot[height, width](m)
    vec._del_old()
