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

# A closure that captures nothing but whose signature names a local's
# immutable origin (here through `p.origin`) should need no capture list, and
# passing it next to a value carrying that origin should not alias.


# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


def consume[T: AnyType, F: def(T)](x: T, f: F):
    f(x)


# TODO(MOCO-4969):
#
# def no_capture_list():
#     var a = 0
#     var p = Pointer(to=a).as_imm()

#     def h(e: Pointer[Int, p.origin]):
#         pass

#     consume(p, h)


# CHECK: lit.struct.decl @"closure$has_capture_list()::h::__storage"
# CHECK-SAME: <["a`"]*"a`": origin<false>
def has_capture_list():
    var a = 0
    var p = Pointer(to=a).as_imm()

    var b = 0

    def h(e: Pointer[Int, p.origin]) {imm b}:
        pass

    consume(p, h)
