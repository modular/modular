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
# RUN: %mojo %s | FileCheck %s

# COM: A closure that copy-captures a struct whose `RegisterPassable`
# COM: conformance is conditional (`Tuple[*Ts]`) must be `RegisterPassable`
# COM: itself whenever the capture's bindings prove the conformance. The
# COM: declaration-time convention of such a struct is MemoryOnly, so the
# COM: closure emitter has to evaluate the constraint per instantiation.


def launch[F: RegisterPassable & def() -> Int](f: F) -> Int:
    return f()


def concrete() -> Int:
    var t = Tuple(Int32(7), Int32(9))

    def body() {var t} -> Int:
        return Int(t[0]) + Int(t[1])

    comptime assert conforms_to(type_of(body), RegisterPassable)
    return launch(body)


def generic[T: TrivialRegisterPassable](a: T, b: T) -> Int:
    var t = Tuple(a, b)

    def body() {var t} -> Int:
        return 2

    comptime assert conforms_to(type_of(body), RegisterPassable)
    return launch(body)


def refined[T: ImplicitlyCopyable & Deinitable](a: T, b: T) -> Int:
    var t = Tuple(a, b)
    comptime assert conforms_to(type_of(t), RegisterPassable)

    def body() {var t} -> Int:
        return 3

    comptime assert conforms_to(type_of(body), RegisterPassable)
    return launch(body)


def main():
    # CHECK: 16
    print(concrete())
    # CHECK: 2
    print(generic(Int32(1), Int32(2)))
    # CHECK: 3
    print(refined(Int64(1), Int64(2)))
