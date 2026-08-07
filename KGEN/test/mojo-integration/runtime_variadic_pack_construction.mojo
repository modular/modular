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
#
# Pin: a runtime `VariadicPack` built via `lit.ref.pack.from_pointer_pack`
# can be unpacked into a `def(*args: *Ts) thin` callee while `Ts` is still
# outer-bound — the shape Python-Mojo bindings (MOCO-2210) depend on.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


@fieldwise_init
struct MyObject(Copyable, Writable):
    var value: Int


def sink_two(a: MyObject, b: MyObject):
    print("sink_two:", a, b)


def sink_three(a: MyObject, b: MyObject, c: MyObject):
    print("sink_three:", a, b, c)


def call_from_two_ptrs[
    Ts: TypeList[Trait=AnyType, ...],
    //,
    # fmt: off
    callee: def(*args: *Ts) thin,
    # fmt: on
](mut a0: Ts[0], mut a1: Ts[1]):
    var ptr_tuple = Tuple(Pointer(to=a0), Pointer(to=a1))

    # Use `*Ts` rather than spelling out element types so the pack's
    # parametric variadic matches the callee's parametric variadic.
    comptime PackType = VariadicPack[
        origin=MutAnyOrigin,
        element_trait=AnyType,
        False,
        *Ts,
    ]
    var pack = PackType(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[_type=PackType._mlir_type](
            ptr_tuple._mlir_value
        )
    )
    callee(*pack)


def call_from_three_ptrs[
    Ts: TypeList[Trait=AnyType, ...],
    //,
    # fmt: off
    callee: def(*args: *Ts) thin,
    # fmt: on
](mut a0: Ts[0], mut a1: Ts[1], mut a2: Ts[2]):
    var ptr_tuple = Tuple(Pointer(to=a0), Pointer(to=a1), Pointer(to=a2))
    comptime PackType = VariadicPack[
        origin=MutAnyOrigin,
        element_trait=AnyType,
        False,
        *Ts,
    ]
    var pack = PackType(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[_type=PackType._mlir_type](
            ptr_tuple._mlir_value
        )
    )
    callee(*pack)


def main():
    var x = MyObject(1)
    var y = MyObject(2)
    # CHECK: sink_two: MyObject(value=1) MyObject(value=2)
    call_from_two_ptrs[sink_two](x, y)

    var a = MyObject(3)
    var b = MyObject(4)
    var c = MyObject(5)
    # CHECK: sink_three: MyObject(value=3) MyObject(value=4) MyObject(value=5)
    call_from_three_ptrs[sink_three](a, b, c)
