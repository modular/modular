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
# A thin function pointer whose argument list is a `TypeList.splat[1, T]()`
# variadic pack is conceptually identical to one written with a single
# non-variadic argument of type `T`. This checks that a plain, non-variadic
# function is assignable to both forms, also when keyword-only arguments
# follow the pack.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


struct Memory:
    def __init__(out self):
        pass


def takes_one(arg: Memory):
    print("yes")


def takes_one_with_kwargs(arg: Memory, var **kwargs: Int):
    print(len(kwargs))


def takes_two_then_tag(a: Int, b: Int, *, tag: Int):
    print(a, b, tag)


def takes_all(name: String, a: Int, b: Int, *, tag: Int, var **kwargs: Int):
    print(name, a, b, tag, len(kwargs))


def infers_pack[*Ts: AnyType](f: def(*args: *Ts, tag: Int) thin):
    print("inferred")


def main():
    var f1: def(Memory) thin = takes_one
    var f2: def(*args: *TypeList.splat[1, Memory]()) thin = takes_one
    var f3: def(
        *args: *TypeList.splat[1, Memory](), var **kwargs: Int
    ) thin = takes_one_with_kwargs

    # CHECK: yes
    f1(Memory())
    # CHECK: yes
    f2(Memory())
    # CHECK: 1
    f3(Memory(), value=1)

    # Keyword-only arguments can follow the pack. They match the arguments
    # of the plain function by name.
    var f4: def(
        *args: *TypeList.splat[2, Int](), tag: Int
    ) thin = takes_two_then_tag
    # CHECK: 1 2 3
    f4(1, 2, tag=3)

    var f5: def(
        name: String,
        *args: *TypeList.splat[2, Int](),
        tag: Int,
        var **kwargs: Int,
    ) thin = takes_all
    # CHECK: x 4 5 6 1
    f5("x", 4, 5, tag=6, value=7)

    # CHECK: inferred
    infers_pack(takes_two_then_tag)
