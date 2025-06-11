# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import time
from random import random_si64, seed
from sys.info import sizeof

import microbenchmark
from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.matmul import matmul, pack_b_ndbuffer, pack_matmul_b_shape_func
from memory import UnsafePointer, bitcast
from microbenchmark import Benchmarkable

from utils.index import Index, IndexList

alias alignment = 64


fn gemm_naive[
    a_type: DType, b_type: DType, c_type: DType
](
    a: NDBuffer[a_type, 2, _, DimList.create_unknown[2]()],
    b: NDBuffer[b_type, 2, _, DimList.create_unknown[2]()],
    c: NDBuffer[c_type, 2, _, DimList.create_unknown[2]()],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c_type]()
                var b_val = b[p, j].cast[c_type]()
                c[i, j] += a_val * b_val


@fieldwise_init
struct MatmulNaiveTest[a_type: DType, b_type: DType, c_type: DType](
    Benchmarkable, Copyable, Movable
):
    var m: Int
    var n: Int
    var k: Int
    var a_ptr: UnsafePointer[Scalar[a_type]]
    var b_ptr: UnsafePointer[Scalar[b_type]]
    var c_ptr: UnsafePointer[Scalar[c_type]]
    var am: NDBuffer[a_type, 2, MutableAnyOrigin, DimList.create_unknown[2]()]
    var bm: NDBuffer[b_type, 2, MutableAnyOrigin, DimList.create_unknown[2]()]
    var cm: NDBuffer[c_type, 2, MutableAnyOrigin, DimList.create_unknown[2]()]

    fn __init__(out self, m: Int, n: Int, k: Int):
        self.m = m
        self.n = n
        self.k = k
        self.a_ptr = UnsafePointer[Scalar[a_type]].alloc(
            m * k, alignment=alignment
        )
        self.b_ptr = UnsafePointer[Scalar[b_type]].alloc(
            k * n, alignment=alignment
        )
        self.c_ptr = UnsafePointer[Scalar[c_type]].alloc(
            m * n, alignment=alignment
        )
        self.am = NDBuffer[a_type, 2, DimList.create_unknown[2]()](
            self.a_ptr, Index(self.m, self.k)
        )
        self.bm = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
            self.b_ptr, Index(self.k, self.n)
        )
        self.cm = NDBuffer[c_type, 2, DimList.create_unknown[2]()](
            self.c_ptr, Index(self.m, self.n)
        )

    @no_inline
    fn __str__(self) -> String:
        return String("m = ", self.m, ", n = ", self.n, ", k = ", self.k)

    fn __del__(owned self):
        self.a_ptr.free()
        self.b_ptr.free()
        self.c_ptr.free()

    @always_inline
    fn global_pre_run(self):
        print("Generating Random Input Data")
        for i in range(self.m):
            for j in range(self.k):
                var val = random_si64(0, 255)
                self.am[i, j] = val.cast[a_type]()
        for i in range(self.k):
            for j in range(self.n):
                var val = random_si64(-128, 127)
                self.bm[i, j] = val.cast[b_type]()
        for i in range(self.m):
            for j in range(self.n):
                self.cm[i, j] = 0

    @always_inline
    fn pre_run(self):
        pass

    fn run(self):
        gemm_naive[a_type, b_type, c_type](
            self.am, self.bm, self.cm, self.m, self.n, self.k
        )

    @always_inline
    fn post_run(self):
        pass

    @always_inline
    fn global_post_run(self):
        pass


@fieldwise_init
struct MatmulTest[a_type: DType, b_type: DType, c_type: DType](
    Benchmarkable, Copyable, Movable
):
    var m: Int
    var n: Int
    var k: Int
    var a_ptr: UnsafePointer[Scalar[a_type]]
    var b_ptr: UnsafePointer[Scalar[b_type]]
    var c_ptr: UnsafePointer[Scalar[c_type]]
    var am: NDBuffer[a_type, 2, DimList.create_unknown[2]()]
    var bm: NDBuffer[b_type, 2, DimList.create_unknown[2]()]
    var cm: NDBuffer[c_type, 2, DimList.create_unknown[2]()]

    fn __init__(out self, m: Int, n: Int, k: Int):
        self.m = m
        self.n = n
        self.k = k
        self.a_ptr = UnsafePointer[Scalar[a_type]].alloc(
            self.m * self.k, alignment=alignment
        )
        self.b_ptr = UnsafePointer[Scalar[b_type]].alloc(
            self.k * self.n, alignment=alignment
        )
        self.c_ptr = UnsafePointer[Scalar[c_type]].alloc(
            self.m * self.n, alignment=alignment
        )
        self.am = NDBuffer[a_type, 2, DimList.create_unknown[2]()](
            self.a_ptr, Index(self.m, self.k)
        )
        self.bm = NDBuffer[b_type, 2, DimList.create_unknown[2]()](
            self.b_ptr, Index(self.k, self.n)
        )
        self.cm = NDBuffer[c_type, 2, DimList.create_unknown[2]()](
            self.c_ptr, Index(self.m, self.n)
        )

    @no_inline
    fn __str__(self) -> String:
        return String("m = ", self.m, ", n = ", self.n, ", k = ", self.k)

    fn __del__(owned self):
        self.a_ptr.free()
        self.b_ptr.free()
        self.c_ptr.free()

    @always_inline
    fn global_pre_run(self):
        print("Generating Random Input Data")
        for i in range(self.m):
            for j in range(self.k):
                var val = random_si64(0, 255)
                self.am[i, j] = val.cast[a_type]()
        for i in range(self.k):
            for j in range(self.n):
                var val = random_si64(-128, 127)
                self.bm[i, j] = val.cast[b_type]()
        for i in range(self.m):
            for j in range(self.n):
                self.cm[i, j] = 0

    @always_inline
    fn pre_run(self):
        pass

    fn run(self):
        matmul[
            a_type,
            DimList.create_unknown[2](),
            b_type,
            DimList.create_unknown[2](),
            c_type,
            DimList.create_unknown[2](),
        ](self.cm.data, self.am.data, self.bm.data)

    @always_inline
    fn post_run(self):
        pass

    @always_inline
    fn global_post_run(self):
        pass


fn main():
    microbenchmark.run(
        MatmulTest[DType.uint8, DType.int8, DType.int32](800, 800, 160),
        "opt_matmul_1",
    )
    microbenchmark.run(
        MatmulTest[DType.uint8, DType.int8, DType.int32](16, 16, 32),
        "opt_matmul_2",
    )
    microbenchmark.run(
        MatmulTest[DType.uint8, DType.int8, DType.int32](32, 32, 64),
        "opt_matmul_3",
    )
    microbenchmark.run(
        MatmulNaiveTest[DType.uint8, DType.int8, DType.int32](8, 8, 16),
        "naive_matmul_1",
    )
    microbenchmark.run(
        MatmulNaiveTest[DType.uint8, DType.int8, DType.int32](16, 16, 32),
        "naive_matmul_2",
    )
    microbenchmark.run(
        MatmulNaiveTest[DType.uint8, DType.int8, DType.int32](32, 32, 64),
        "naive_matmul_3",
    )
