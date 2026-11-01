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
# RUN: kgen %s -elaborate=no-use-parametric-interpreter -verify-diagnostics


# Tests a GPU-function-like case (see FTAGPUF) but this one catches when the
# user hands in something of the wrong type.
# TODO(MOCO-1106): This doesn't catch when the given argument type
# (DeviceBuffer[Int]) mismatches the expected argument type
# (UnsafePointer[Float32]) by only the input parameter (Int vs Float32), that's
# only caught by elaborator for now and is tested elsewhere (search FTAGPUF).


# expected-error @+2{{function instantiation failed}}
@export
def entry_method() abi("Mojo"):
    foo()  # expected-note {{call expansion failed}}


trait ZRPTrait(TrivialRegisterPassable):
    def print(self):
        pass


@fieldwise_init
struct ZInt(ZRPTrait):
    var i: Int

    def print(self):
        print("im an int: ", self.i)


@fieldwise_init
struct ZFloat(ZRPTrait):
    var f: Float32

    def print(self):
        print("im a float: ", self.f)


# Copied from stdlib, so we can catch the error message here.
@always_inline("nodebug")
def my_rebind[
    src_type: TrivialRegisterPassable,
    //,
    dest_type: TrivialRegisterPassable,
](src: src_type) -> dest_type:
    # expected-note @below {{rebind input type '!kgen.scalar<f32>' does not match result type '!kgen.scalar<index>'}}
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


trait ConvertibleToZPointer:
    comptime Pointee: ZRPTrait

    def to_zpointer(self) -> ZPointer[Self.Pointee]:
        ...


struct ZPointer[T: ZRPTrait](GPUPassable):
    var thing: Self.T

    def __init__(out self, thing: Self.T):
        self.thing = thing

    def __init__(out self, other: ZPointer[Self.T]):
        self.thing = other.thing

    @implicit
    # expected-note @below {{function instantiation failed}}
    def __init__[C: ConvertibleToZPointer](out self, c: C):
        # TODO(MOCO-1106): Add a constrained call here for better error message.
        # TODO(MOCO-1106): If we can remove this rebind, we win. We'd need to
        # constrain C.Pointee=T somehow, or make ConvertibleToZPointer into a
        # generic trait instead of using an associated alias.
        # As it is, this won't catch incorrectly passing in a e.g.
        # ZDeviceBuffer[Int] into a ZPointer[Bool].
        # expected-note @below {{call expansion failed}}
        self.thing = my_rebind[ZPointer[Self.T]](c.to_zpointer()).thing
        self.thing.print()


trait ConvertibleToZTileTensor:
    def to_tensor(self) -> ZTileTensor:
        ...


struct ZTileTensor(GPUPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZTileTensor](out self, c: C):
        var `_`: ZTileTensor = c.to_tensor()


trait GPUPassable(TrivialRegisterPassable):
    pass


@fieldwise_init
struct DeviceFunction[*Ts: GPUPassable](ImplicitlyCopyable):
    def call(self, *args: *Self.Ts) -> Int:
        return Int()


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
struct NDBuffer(GPUPassable):
    pass


def compile[
    ArgTypes: TypeList[Trait=GPUPassable, ...],
    //,
    func: def(*args: *ArgTypes) thin -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


# expected-note @+1{{function instantiation failed}}
def foo():
    def kernel(t: ZTileTensor, p: ZPointer[ZInt], n: NDBuffer) -> Int:
        return Int()

    var thing = compile[kernel]()
    var mtt = ManagedTileTensor()
    var wrong_db = DeviceBuffer[ZFloat](ZFloat(73.0))
    var ndb = NDBuffer()
    # Using wrong_db (DeviceBuffer[ZFloat]) here is incorrect, doesn't match the
    # kernel's expected DeviceBuffer[Int].
    # The elaborator should catch this.
    # TODO(MOCO-1106): Make the parser catch this instead.
    # expected-note @below {{call expansion failed}}
    var `_`: Int = thing.call(mtt, wrong_db, ndb)
