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

# Writes through a `mut *args` pack reach the caller when the callee is not
# inlined, including when keyword-only arguments follow the pack and when the
# pack is forwarded by name (MOCO-5272).

# RUN: mojo %s | FileCheck %s --check-prefix=OUT

# The IR dump is produced only when the pass pipeline actually runs, and a
# warm KGEN transform cache serves the result without running it, so give
# the run an empty cache of its own. With one thread, the dump shows the
# functions in a fixed order.
# RUN: rm -rf %t.cache && env MODULAR_CACHE_DIR=%t.cache kgen --elaborate %s \
# RUN:   -mllvm -mlir-disable-threading \
# RUN:   -mllvm --mlir-print-ir-before=lower-arg-conventions -o /dev/null 2>&1 \
# RUN:   | FileCheck %s --check-prefix=IR

comptime Elt = Defaultable & Movable & Deinitable


# The pack vararg and its element convention survive to argument lowering.
# IR: kgen.func @"variadic_pack_mut_outlined::zero_all[{{.*}} pack_mut)
@no_inline
def zero_all[*Ts: Elt](mut *args: *Ts):
    comptime for i in range(args.__len__()):
        args[i] = Ts[i]()


# IR: kgen.func @"variadic_pack_mut_outlined::zero_all_kw[{{.*}} pack_mut)
@no_inline
def zero_all_kw[*Ts: Elt](mut *args: *Ts, tag: Int = 0):
    comptime for i in range(args.__len__()):
        args[i] = Ts[i]()


@no_inline
def zero_through[
    o: MutOrigin, *Ts: Elt
](pack: VariadicPack[origin=o, element_trait=Elt, False, *Ts]):
    comptime for i in range(pack.__len__()):
        pack[i] = Ts[i]()


@no_inline
def forward_by_name[*Ts: Elt](mut *args: *Ts):
    zero_through(args)


# This struct has one field, so LowerLIT flattens it to the value pack of the
# tuple. A pack element that is a value pack is not split again.
@fieldwise_init
struct TupleBox(Defaultable, Movable):
    var t: Tuple[Int, Int]

    def __init__(out self):
        self.t = (0, 0)


def main():
    var a = 1
    var b = 2
    zero_all(a, b)
    # OUT: zero_all: 0 0
    print("zero_all:", a, b)

    a = 1
    b = 2
    zero_all_kw(a, b, tag=7)
    # OUT: zero_all_kw: 0 0
    print("zero_all_kw:", a, b)

    a = 1
    b = 2
    forward_by_name(a, b)
    # OUT: forward_by_name: 0 0
    print("forward_by_name:", a, b)

    var x = TupleBox((1, 2))
    var y = TupleBox((3, 4))
    zero_all(x, y)
    # OUT: value pack elements: 0 0 0 0
    print("value pack elements:", x.t[0], x.t[1], y.t[0], y.t[1])

    a = 1
    b = 2
    var f = zero_all[Int, Int]
    f(a, b)
    # OUT: function pointer: 0 0
    print("function pointer:", a, b)
