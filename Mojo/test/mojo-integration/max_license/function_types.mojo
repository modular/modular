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
# Various integration tests for function types.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

from std.os import abort


# ===----------------------------------------------------------------------=== #
# GPU Function example
# ===----------------------------------------------------------------------=== #

# Tests a more advanced GPU-function-like case (see FTGPUFT), end-to-end.
# Specifically, it checks that the entire pipeline will allow the proper
# argument types through.
# This, paired with the _errors test, makes sure GPU function works end-to-end.
# These are here as integration tests because some kinds of errors we can only
# catch at elaboration time, because of the rebind below.
# TODO(MOCO-1106): Move this detection completely to the parser, and remove
# these integration tests.


trait ZRPTrait(TrivialRegisterPassable):
    def print(self):
        pass


@fieldwise_init
struct ZInt(ZRPTrait):
    var i: Int

    def print(self):
        print("im an int:", self.i)


trait ConvertibleToZPointer:
    comptime Pointee: ZRPTrait

    def to_zpointer(self) -> ZPointer[Self.Pointee]:
        ...


struct ZPointer[T: ZRPTrait](GPUPassable, ZRPTrait):
    var thing: Self.T

    def __init__(out self, thing: Self.T):
        self.thing = thing

    def __init__(out self, other: ZPointer[Self.T]):
        self.thing = other.thing

    @implicit
    def __init__[C: ConvertibleToZPointer](out self, c: C):
        # TODO(MOCO-1106): If we can remove this rebind, we win. We'd need to
        # constrain C.Pointee=T somehow, or make ConvertibleToZPointer into a
        # generic trait instead of using an associated alias.
        # As it is, this won't catch incorrectly passing in a e.g.
        # ZDeviceBuffer[Int] into a ZPointer[Bool].
        var other: ZPointer[Self.T] = rebind[ZPointer[Self.T]](c.to_zpointer())
        self.thing = other.thing
        self.thing.print()

    def print(self):
        print("i am a pointer")


trait ConvertibleToZTileTensor:
    def to_tensor(self) -> ZTileTensor:
        ...


struct ZTileTensor(GPUPassable, ZRPTrait):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZTileTensor](out self, c: C):
        var z: ZTileTensor = c.to_tensor()
        _ = z

    def print(self):
        print("i am a layout tensor")


trait GPUPassable:
    pass


@fieldwise_init
struct DeviceFunction[*Ts: GPUPassable](ImplicitlyCopyable):
    def call(self, *args: *Self.Ts) -> Int:
        return 73


struct DeviceBuffer[T: ZRPTrait](ConvertibleToZPointer):
    comptime Pointee: ZRPTrait = Self.T
    var thing: Self.T

    def __init__(out self, thing: Self.T):
        self.thing = thing

    def to_zpointer(self) -> ZPointer[Self.T]:
        return ZPointer[Self.T](self.thing)


@fieldwise_init
struct ManagedTileTensor(ConvertibleToZTileTensor, ImplicitlyCopyable):
    def to_tensor(self) -> ZTileTensor:
        return ZTileTensor()


# Never converted, the GPU just uses this one directly
@fieldwise_init
struct NDBuffer(GPUPassable, TrivialRegisterPassable):
    pass


# Memory type, not register-passable
@fieldwise_init
struct MemBuffer(GPUPassable):
    pass


def compile[
    ArgTypes: TypeList[Trait=GPUPassable, ...],
    //,
    func: def(*args: *ArgTypes) thin -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


def test_call():
    def kernel(
        t: ZTileTensor, p: ZPointer[ZInt], n: NDBuffer, mb: MemBuffer
    ) -> Int:
        return Int()

    var thing = compile[kernel]()
    var mtt = ManagedTileTensor()
    var db = DeviceBuffer[ZInt](ZInt(42))
    var ndb = NDBuffer()
    var mb = MemBuffer()
    # All the argument types line up nicely.
    # CHECK: im an int: 42
    _ = thing.call(mtt, db, ndb, mb)


# ===----------------------------------------------------------------------=== #
# Function type convertibility.
# ===----------------------------------------------------------------------=== #


def fn_doesnt_raise() -> Int:
    return 42


struct ImplicitFromInt:
    var value: Float32

    @implicit
    def __init__(out self, int: Int):
        self.value = Float32(int)


def test_more_conversions():
    print("test_more_conversions")
    # CHECK: test_more_conversions

    var test_raises: def() thin raises -> Int = fn_doesnt_raise
    var test_result_convert: def() thin raises -> ImplicitFromInt = (
        fn_doesnt_raise
    )

    try:
        # CHECK: use as raising: 42
        print("use as raising:", test_raises())

        # CHECK: use as raising float: 42.0
        print("use as raising float:", test_result_convert().value)
    except e:
        # Not reached.
        abort()


def main():
    test_call()
    test_more_conversions()
