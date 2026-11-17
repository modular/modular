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
# Tests the interop/passing-data-to-mojo.mdx examples: a C program calls the
# exported functions from a `--emit shared-lib` library. The page's Mojo
# blocks appear verbatim; `stats_add()` and the string exports fill in what
# the page describes in prose.
#
# Page claims, and where each is verified (CHECK lines below):
#   1. `stats_new()` hands out a handle; other exports use it; `stats_free()`
#      rebuilds the `Allocation` and frees it                -> stats
#   2. Exported functions take and return C structs by value, and `Point`
#      needs no particular trait                 -> distance, point order
#   2a. An exported function takes a struct with an `Array` field by value,
#       and the struct needs no particular trait               -> label
#   2b. An exported function returns that struct by value, although
#       `Array` keeps it from being `RegisterPassable` (data-types.mdx)
#                                                            -> made label
#   3. An export views a caller's array as a `Span`, and the caller skips
#      the call for an empty array, so no null reaches `Pointer` -> rms
#   4. An export takes a `const char*` the caller owns, returns a new string
#      from `unsafe_leak()`, and a matching export frees it by counting to
#      the zero byte                                         -> greet
#   5. An export calls a C callback with the caller's context, which comes
#      back unchanged                                        -> integrate_sin
#
# Not tested:
#   - The C++ (`unique_ptr`, `std::vector`) and Rust callers: they make the
#     same C ABI calls that c_host.c makes.
#   - An export that raises, foreign `free()` on Mojo memory, and mismatched
#     struct or callback declarations: compile errors or undefined behavior.
#
# COM: Uses `.dylib` on every platform, as interoperability/ does. dlopen
# COM:   takes a literal path, so the extension is cosmetic.
# RUN: mkdir -p %t.dir
# RUN: %mojo-build %s -o %t.dir/libexports.dylib --emit shared-lib
# RUN: $(dirname %s)/c_host %t.dir/libexports.dylib | FileCheck %s

# CHECK: stats mean = 3.0000
# CHECK-NEXT: distance = 5.0000
# CHECK-NEXT: point order = 1234
# CHECK-NEXT: label = 707
# CHECK-NEXT: made label = 9 sensor9
# CHECK-NEXT: rms = 3.5355
# CHECK-NEXT: greet = Hello, Mojo!
# CHECK-NEXT: 25%
# CHECK-NEXT: 50%
# CHECK-NEXT: 75%
# CHECK-NEXT: 100%
# CHECK-NEXT: area = 2.0523, calls = 4

from std.ffi import (
    CStringSpan,
    c_char,
    c_double,
    c_int,
    c_size_t,
    external_call,
)


# start-handles
from std.memory.alloc import alloc, dealloc, Allocation, Layout


@fieldwise_init
struct Stats:
    var count: Int
    var mean: Float64


comptime Handle = MutOpaquePointer[MutUntrackedOrigin]


@export("stats_new")
def stats_new() abi("C") -> Handle:
    var stats = alloc(Layout[Stats].single()).unsafe_leak()
    stats.write(Stats(count=0, mean=0.0))
    return stats.unsafe_bitcast[NoneType]()


@export("stats_free")
def stats_free(handle: Handle) abi("C"):
    dealloc(
        Allocation(
            unsafe_owned_ptr=handle.unsafe_bitcast[Stats](),
            layout=Layout[Stats].single(),
        )
    )


# end-handles


@export("stats_add")
def stats_add(handle: Handle, value: c_double) abi("C") -> c_double:
    var stats = handle.unsafe_bitcast[Stats]()
    stats[].count += 1
    stats[].mean += (value - stats[].mean) / Float64(stats[].count)
    return stats[].mean


# start-structs
from std.math import sqrt


@fieldwise_init
struct Point:
    var x: Float64
    var y: Float64


@export("distance")
def distance(a: Point, b: Point) abi("C") -> Float64:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# end-structs


# `distance()` gives the same result with x and y swapped, or a and b
# swapped, so this export pins the field and argument order.
@export("point_order")
def point_order(a: Point, b: Point) abi("C") -> Float64:
    return a.x * 1000 + a.y * 100 + b.x * 10 + b.y


# A C struct with an array field: struct label { int id; char name[16]; }.
# `Array` isn't `RegisterPassable`, so neither is `Label`, and it still
# crosses by value.
@fieldwise_init
struct Label(Movable):
    var id: c_int
    var name: Array[c_char, 16]


@export("label_id_plus_length")
def label_id_plus_length(label: Label) abi("C") -> c_int:
    var length = Int(external_call["strlen", c_size_t](label.name.unsafe_ptr()))
    return label.id * 100 + c_int(length)


@export("make_label")
def make_label(id: c_int) abi("C") -> Label:
    var name = Array[c_char, 16](fill=0)
    for index, byte in enumerate("sensor".as_bytes()):
        name[index] = c_char(byte)
    name[6] = c_char(ord("0")) + c_char(id)
    return Label(id=id, name=name^)


# start-arrays
from std.ffi import c_size_t
from std.math import sqrt


@export("rms")
def rms(
    samples: Pointer[Float32, ImmUntrackedOrigin], count: c_size_t
) abi("C") -> Float64:
    var total: Float64 = 0
    for sample in Span(unsafe_ptr=samples, length=Int(count)):
        total += Float64(sample) ** 2
    return sqrt(total / Float64(count))


# end-arrays


# The Strings section is prose. These two exports follow it: copy the
# caller's string, return a new one from `unsafe_leak()`, and free it by
# counting up to the terminating zero.
@export("greet")
def greet(
    name: Pointer[c_char, ImmUntrackedOrigin]
) abi("C") -> Pointer[c_char, MutUntrackedOrigin]:
    # An export can't raise, so the lossy conversion stands in for a check.
    var span = CStringSpan(unsafe_from_ptr=name)
    var text = "Hello, " + String(from_utf8_lossy=span.as_bytes()) + "!"
    var length = text.byte_length()
    var copy = alloc(Layout[c_char](length=length + 1)).unsafe_leak()
    var bytes = text.as_bytes()
    for i in range(length):
        copy[unsafe_offset=i] = c_char(bytes[i])
    copy[unsafe_offset=length] = 0
    return copy


@export("free_string")
def free_string(string: Pointer[c_char, MutUntrackedOrigin]) abi("C"):
    var length = Int(external_call["strlen", c_size_t](string))
    dealloc(
        Allocation(
            unsafe_owned_ptr=string, layout=Layout[c_char](length=length + 1)
        )
    )


# start-callbacks
from std.ffi import c_double, c_int
from std.math import pi, sin

comptime Context = MutOpaquePointer[MutUntrackedOrigin]
comptime Progress = def(c_double, Context) thin abi("C")


@export("integrate_sin")
def integrate_sin(
    steps: c_int, progress: Progress, context: Context
) abi("C") -> c_double:
    var width = pi / Float64(steps)
    var area: Float64 = 0
    for i in range(Int(steps)):
        area += sin((Float64(i) + 0.5) * width) * width
        progress(Float64(i + 1) / Float64(steps), context)
    return area


# end-callbacks
