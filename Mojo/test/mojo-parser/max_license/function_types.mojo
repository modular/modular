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
# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s


@fieldwise_init
struct MemType(ImplicitlyCopyable):
    pass


struct ParamType[p: Int](TrivialRegisterPassable):
    pass


def generic_func[T: AnyType, U: AnyType](x: T, y: U) -> T:
    pass


def generic_func[T: AnyType](x: T):
    pass


def overloaded() -> Int:
    pass


def overloaded(x: Int):
    pass


@fieldwise_init
struct RegParam[a: Int](ImplicitlyCopyable, RegisterPassable):
    def foo[b: Int](self, c: Int) -> Self:
        return self

    def baz[x: Int, y: ParamType[x]](self):
        pass

    def shadow[b: Int](self):
        pass

    def bar[x: Int, y: Int](self, z: ParamType[x]) -> ParamType[y]:
        pass


def unbound_foo[T: AnyType, b: Int](self: T, c: Int) -> T:
    pass


def unbound_baz[T: AnyType, x: Int, y: ParamType[x]](self: T):
    pass


def unbound_shadow[T: AnyType, b: Int](self: T):
    pass


def unbound_bar[
    T: AnyType, x: Int, y: Int
](self: T, z: ParamType[x]) -> ParamType[y]:
    pass


def take[T: AnyType](var x: T, var y: Int):
    pass


def take_int(var x: Int, var y: Int):
    pass


__async def async_generic[T: AnyType](self: T) -> Int:
    pass


__async def specific_async(self: Int) -> Int:
    pass


# CHECK-LABEL: lit.fn @"test_conversion
def test_conversion[p: Int](x: Int):
    # CHECK-NEXT: = <rebind(:{{.*}}[[CONVERTER1:@.*]]<:!AnyType ?, {{.*}}rebind({{.*}}generic_func{{.*}}<:!AnyType ?, :!AnyType !Int{{.*}}>
    comptime f1: def[X: AnyType](X, Int) thin -> X = generic_func[U=Int, ...]


struct defAdapter[In: AnyType, Out: AnyType](Movable where False):
    var func: def(Self.In) thin -> Self.Out

    def __init__(out self: Self, f: def(Self.In) thin -> Self.Out):
        self.func = f


def mem_to_int(value: MemType) -> Int:
    return 43


def int_to_mem(value: Int) -> MemType:
    return MemType()


def int_to_int(value: Int) -> Int:
    return value


def mem_to_mem(value: MemType) -> MemType:
    return value


def test_fn_adapter():
    var str_to_int = defAdapter(mem_to_int)
    var int_to_str = defAdapter(int_to_mem)
    var int_to_int = defAdapter(int_to_int)
    var string_to_string = defAdapter(mem_to_mem)


struct DependentParam[a: Int, //, b: RegParam[a]](RegisterPassable):
    pass


def crazy_signature[
    T: AnyType, a: Int, b: Int, p: RegParam[a]
](c: DependentParam[p], x: T) -> RegParam[b]:
    pass


def test_crazy_bind[a: Int, b: Int]():
    comptime f: def[p: RegParam[a]](DependentParam[p], Int) thin -> RegParam[
        b
    ] = crazy_signature[Int, a, b, _]


def take_mut(mut x: Int):
    pass


def test_mut_fp():
    var a: def(mut x: Int) thin -> None = take_mut
    # Shouldn't need arg name.
    var b: def(mut: Int) thin -> None = take_mut
    var c: def(Int) thin -> Int = int_to_int
    # thin and abi("C") can appear together on a function type, in either order.
    var d: def(Int) thin abi("C") -> Int
    var e: def(Int) abi("C") thin -> Int


# // -----


struct ParamType[p: Int](TrivialRegisterPassable):
    pass


def generic_func[T: AnyType, U: AnyType](x: T, y: U) -> T:
    pass


# CHECK-LABEL: lit.fn @"test_conversion
def test_conversion[p: Int](x: Int):
    # CHECK-NEXT: <<"X": !AnyType>[1](!lit.struct<#ParamType <:!Int p>>, !lit.ref<:!AnyType *(0,0), imm *[0,0]> imm_mem, |) -> !lit.struct<#ParamType <:!Int p>>
    # CHECK-SAME: [[CONVERTER3:@.*]]<:!Int p, :!AnyType ?, :
    comptime f4: def[X: AnyType](ParamType[p], X) thin -> ParamType[
        p
    ] = generic_func[ParamType[p], ...]


# CHECK: lit.fn {{.*}}<[""]_0: !Int, [""]_1: !AnyType, [""]callee: {{.*}}>[{{.*}}](%0[*""]: !lit.struct<#ParamType <:!Int _0>>
# CHECK-SAME: %1[*""]: !lit.ref<:!AnyType _1,


# // -----


def raises_func[T: AnyType](x: T) raises:
    pass


# CHECK-LABEL: lit.fn @"test_conversion
def test_conversion[p: Int](x: Int):
    # CHECK: f5{{.*}}def(::SIMD[DType.int, 1]) raises thin -> None|def(x: ::SIMD[DType.int, 1]) raises thin -> None
    comptime f5: def(Int) thin raises -> None = raises_func[Int]


# CHECK: lit.fn @"def(::SIMD[DType.int, 1]) raises thin -> None|def(x: ::SIMD[DType.int, 1]) raises thin -> None
# CHECK: [[TMP:%.*]] = lit.ref.immut
# CHECK: call{{.*}}([[TMP]], %1, %2)


# // -----


def generic_func[T: AnyType, U: AnyType](x: T, y: U) -> T:
    pass


# CHECK-LABEL: lit.fn @"test_conversion
def test_conversion[p: Int](x: Int):
    # CHECK-NEXT: = <rebind(:{{.*}}[[CONVERTER0:@.*]]<:!AnyType ?, {{.*}}rebind({{.*}}generic_func{{.*}}<:!AnyType !Int{{.*}}, :!AnyType ?>
    comptime f0: def[X: AnyType](Int, X) thin -> Int = generic_func[Int, ...]


# CHECK: lit.fn [[CONVERTER0]]<[""]_0: !AnyType, [""]callee: !lit.generator<<!AnyType>[3](
# CHECK-SAME: !lit.ref<!Int, imm *[0,0]> imm_mem,
# CHECK-SAME: !lit.ref<:!AnyType *(0,0), imm *[0,1]> imm_mem,
# CHECK-SAME: ?, "__result__": !lit.ref<!Int, mut *[0,2]> byref_result) -> !kgen.none>
# CHECK-SAME: [imm [[LT:\*"0_unnamed`"]]](
# CHECK-SAME: %0[*""]: !Int, %1[*""]: !lit.ref<:!AnyType _0, imm [[LT]]> imm_mem) -> !Int always_inline_no_debug
# CHECK-SAME: thunkKey
# CHECK-NEXT:   [[ANON:%.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!Int,
# CHECK-NEXT:   lit.ref.store %0, [[ANON]]
# CHECK-NEXT:   [[IMM:%.*]] = lit.ref.immut [[ANON]]
# CHECK-NEXT:   [[RESULT:%.*]] = lit.var.decl "__call_result_tmp__"
# CHECK-NEXT:   lit.call[{{.*}}bind_params(:{{.*}} callee, :!AnyType _0)][{{.*}}]([[IMM]], %1, [[RESULT]])
# CHECK-NEXT:   [[REG:%.*]] = lit.load.consume [[RESULT]]
# CHECK-NEXT:   return [[REG]]


# // -----


struct MemType:
    pass


def read_ship_fn(imm x: MemType):
    ...


# We can convert from def(imm MemType)->None to def(mut MemType)->None but not
# vice versa (see TTSMFS).
comptime mut_ship_fn_alias: def(mut MemType) thin -> None = read_ship_fn

# Make sure the thunk was generated
# CHECK-LABEL: lit.fn @"def(mut function_types::MemType) thin -> None|def(x: function_types::MemType) thin -> None|
# CHECK: {{.*}} = lit.call tail[{{.*}}callee][muttoimm {{.*}}](


# // -----


# `imm` marks an argument as an immutable reference; it supports the same
# implicit conversion to `mut` (via a thunk) as the explicit convention above.

struct MemType:
    pass


def imm_ship_fn(imm x: MemType):
    ...


comptime mut_ship_fn_alias_from_imm: def(mut MemType) thin -> None = imm_ship_fn

# Make sure the thunk was generated
# CHECK-LABEL: lit.fn @"def(mut function_types::MemType) thin -> None|def(x: function_types::MemType) thin -> None|
# CHECK: {{.*}} = lit.call tail[{{.*}}callee][muttoimm {{.*}}](


# // -----


# Tests that we can infer a trait input-parameter (T here) from a
# caller-supplied def-parameter-value's (device_func) args (Int), see FSWSTA.


def infer_trait_T[T: AnyType, //, func: def(x: T) thin -> None]():
    pass


def device_func(i: Int):
    pass


# CHECK-LABEL: lit.fn @"test_infer_trait_T_to_Int()"
def test_infer_trait_T_to_Int():
    # CHECK: lit.call {{.*}}@"infer_trait_T{{.*}}"<:!AnyType !Int,
    infer_trait_T[device_func]()


# // -----


# Tests that we can infer the ArgTypes variadic parameter-value from an incoming
# function parameter-value's args (see IFAIAV).


def infer_variadic[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * ArgTypes) thin -> None,
]():
    pass


def device_func(i: Int, b: Bool):
    pass


# CHECK-LABEL: lit.fn @"test_infer_variadic()"
def test_infer_variadic():
    # CHECK: lit.call {{.*}}@"infer_variadic{{[^"]*}}"<
    # CHECK-SAME: :param_list<!AnyType> [!Int, !Bool]
    # CHECK-SAME: :!lit.generator<{{.*}}[1]("args": !lit.ref<!lit.struct<#VariadicPack <:!Bool {:scalar<bool> false}, :origin<false> *(0,0), :!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> *(0,0)>> *(0,1), :meta<!AnyType> !AnyType, :param_list<!AnyType> [!Int, !Bool], :!Bool {:scalar<bool> false}, :!lit.struct<#TypeList <:meta<!AnyType> !AnyType, :param_list<!AnyType> [!Int, !Bool]>> {}>>, imm *[0,0]> imm_mem|pack_vararg)
    infer_variadic[func=device_func]()


def infer_raising_variadic[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *ArgTypes) raises thin -> None,
]():
    pass


def raising_device_func(i: Int, b: Bool) raises:
    pass


# CHECK-LABEL: lit.fn @"test_infer_raising_variadic()"
def test_infer_raising_variadic():
    # CHECK: lit.call {{.*}}@"infer_raising_variadic{{[^"]*}}"<
    # CHECK-SAME: :param_list<!AnyType> [!Int, !Bool]
    infer_raising_variadic[func=raising_device_func]()


def infer_raising_variadic_kwargs[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    func: def(
        *args: *ArgTypes, var **kwargs: Int
    ) raises thin -> None,
]():
    pass


def raising_kwargs_device_func(
    i: Int, b: Bool, var **kwargs: Int
) raises:
    pass


# CHECK-LABEL: lit.fn @"test_infer_raising_variadic_kwargs()"
def test_infer_raising_variadic_kwargs():
    # CHECK: lit.call {{.*}}@"infer_raising_variadic_kwargs{{[^"]*}}"<
    # CHECK-SAME: :param_list<!AnyType> [!Int, !Bool]
    infer_raising_variadic_kwargs[func=raising_kwargs_device_func]()


def accepts_generic_variadic_return[
    ArgTypes: TypeList[Trait=AnyType, ...],
    RetType: AnyType,
    //,
    callee: def(* args: * ArgTypes) thin -> RetType,
]():
    pass


def forwards_none_returning_variadic[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    callee: def(* args: * ArgTypes) thin,
]():
    accepts_generic_variadic_return[callee]()


def none_returning(value: Int):
    _ = value


# Related to MOCO-4631.
# CHECK-LABEL: lit.fn @"test_forward_none_returning_variadic()"
def test_forward_none_returning_variadic():
    # CHECK: lit.call {{.*}}@"forwards_none_returning_variadic{{[^"]*}}"<
    forwards_none_returning_variadic[none_returning]()


# // -----


# More complicated tests that we can correctly map argument positions when doing
# function conversions between functions with different number of arguments
# (because variadics are involved).


def infer_trailing_variadic[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    func: def(a: Int, b: Bool, * args: * ArgTypes) thin -> None,
]():
    pass


def device_func_1(i: Int, b: Bool):
    pass


# CHECK-LABEL: lit.fn @"test_infer_empty_trailing_variadic()"
def test_infer_empty_trailing_variadic():
    # CHECK: lit.call {{.*}}@"infer_trailing_variadic{{[^"]*}}"<
    # CHECK-SAME: :param_list<!AnyType> []
    # CHECK-SAME: :!lit.generator<{{.*}}[1]("a": !Int, "b": !Bool, "args": !lit.ref<!lit.struct<#VariadicPack <:!Bool {:scalar<bool> false}, :origin<false> *(0,0), :!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> *(0,0)>> *(0,1), :meta<!AnyType> !AnyType, :param_list<!AnyType> [], :!Bool {:scalar<bool> false}, :!lit.struct<#TypeList <:meta<!AnyType> !AnyType, :param_list<!AnyType> []>> {}>>, imm *[0,0]> imm_mem|pack_vararg)
    infer_trailing_variadic[device_func_1]()


def device_func_2(i: Int, b: Bool, x: Int, y: Int, z: Int):
    pass


# CHECK-LABEL: lit.fn @"test_infer_nonempty_trailing_variadic()"
def test_infer_nonempty_trailing_variadic():
    # CHECK: lit.call {{.*}}@"infer_trailing_variadic{{[^"]*}}"<
    # CHECK-NOT: :param_list<!AnyType> []
    # CHECK-SAME: :param_list<!AnyType> [!Int, !Int, !Int]
    # CHECK-NOT: :param_list<!AnyType> []
    # CHECK-SAME: :!lit.generator<{{.*}}[1]("a": !Int, "b": !Bool, "args": !lit.ref<!lit.struct<#VariadicPack <:!Bool {:scalar<bool> false}, :origin<false> *(0,0), :!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> *(0,0)>> *(0,1), :meta<!AnyType> !AnyType, :param_list<!AnyType> [!Int, !Int, !Int], :!Bool {:scalar<bool> false}, :!lit.struct<#TypeList <:meta<!AnyType> !AnyType, :param_list<!AnyType> [!Int, !Int, !Int]>> {}>>, imm *[0,0]> imm_mem|pack_vararg)
    infer_trailing_variadic[device_func_2]()


# // -----

# Capture Arguments Into A Struct's Variadic (CAIASV)
# In this test, `compile` captures a given `func` parameter-value's arguments
# into an `ArgTypes` `kgen.variadic` (see IFAIAV), and then returns a struct
# that holds it as an input-parameter of its own.


def device_func(a: Int, b: Bool) -> Int:
    return 73


@fieldwise_init
struct DeviceFunction[*ArgTypes: TrivialRegisterPassable]:
    def call(self, *args: * Self.ArgTypes) -> Int:
        return 91


def compile[
    ArgTypes: TypeList[Trait=TrivialRegisterPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


def main():
    var thing = compile[device_func]()
    var result1 = thing.call(42, True)


# // -----


# Tests that a function can hand a `def` param-ref to a struct's `def` param-decl,
# and inference correctly matches the `def`s against each other.


trait ZAnyRPTrait(TrivialRegisterPassable):
    pass


@fieldwise_init
struct DeviceFunction[
    Ts: TypeList[Trait=ZAnyRPTrait, ...],
    //,
    func: def(* args: * Ts) thin -> Int,
]:
    pass


def compile[
    ArgTypes: TypeList[Trait=ZAnyRPTrait, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> DeviceFunction[func]:
    pass


# // -----


# "GPU Function test": Tests that a callsite infers the variadic from the kernel
# function, and then allows/forces the incoming argument values to be implicitly
# converted to the right types.


trait ConvertibleToZInt:
    def to_zint(self) -> ZInt:
        ...


struct ZInt(ZAnyRPTrait):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZInt](out self, c: C):
        var z: ZInt = c.to_zint()


trait ZAnyRPTrait(TrivialRegisterPassable):
    pass


def device_func(a: ZInt) -> Int:
    return 73


@fieldwise_init
struct DeviceFunction[*ArgTypes: ZAnyRPTrait]:
    def call(self, *args: * Self.ArgTypes) -> Int:
        return 91


@fieldwise_init
struct OriginalStruct(ConvertibleToZInt, Movable where False):
    def to_zint(self) -> ZInt:
        return ZInt()


def compile[
    ArgTypes: TypeList[Trait=ZAnyRPTrait, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


def main():
    var thing = compile[device_func]()
    var s = OriginalStruct()
    var result1 = thing.call(s)


# // -----

# Test FTAGPUF: Tests an advanced GPU-function-like case.
# More complicated version of the GPU function test (see FTGPUFT). Differences:
# - It has three arguments
# - It implicitly converts two arguments
# - One of them is generic (ZPointer)
# - The third argument (NDBuffer) needs no conversion
# - We also have a return type for the kernel because why not.
# - Uses implicit conformance (mostly because I'm not sure how to use explicit
#   conformance for the ZPointer one).
# TODO(MOCO-1106): This catches some problems at parse-time, but it uses a
# rebind below in one place, which only catches things at elaboration time.
# There are some tests elsewhere like this but that show it correctly catching
# incorrect calls, search FTAGPUF.


trait ConvertibleToZPointer:
    comptime Pointee: AnyType

    def to_zpointer(self) -> ZPointer[Self.Pointee]:
        ...


struct ZPointer[T: AnyType](GPUPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZPointer](out self, c: C):
        # TODO(MOCO-1106): If we can remove this rebind, we win. We'd need to
        # constrain C.Pointee=T somehow, or make ConvertibleToZPointer into a
        # generic trait instead of using an associated alias.
        # As it is, this won't catch incorrectly passing in a e.g.
        # ZDeviceBuffer[Int] into a ZPointer[Bool].
        var z: ZPointer[Self.T] = rebind[ZPointer[Self.T]](
            c.to_zpointer()
        )


trait ConvertibleToZTileTensor:
    def to_tensor(self) -> ZTileTensor:
        ...


struct ZTileTensor(GPUPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZTileTensor](out self, c: C):
        var z: ZTileTensor = c.to_tensor()


trait GPUPassable(TrivialRegisterPassable):
    pass


@fieldwise_init
struct DeviceFunction[*ArgTypes: GPUPassable]:
    def call(self, *args: * Self.ArgTypes) -> Int:
        return 91


@fieldwise_init
struct DeviceBuffer[T: AnyType](ConvertibleToZPointer, Movable where False):
    comptime Pointee: AnyType = Self.T

    def to_zpointer(self) -> ZPointer[Self.T]:
        return ZPointer[Self.T]()


@fieldwise_init
struct ManagedTileTensor(ConvertibleToZTileTensor, Movable where False):
    def to_tensor(self) -> ZTileTensor:
        return ZTileTensor()


# Never converted, the GPU just uses this one directly
@fieldwise_init
struct NDBuffer(GPUPassable):
    pass


def kernel(t: ZTileTensor, p: ZPointer[Int], n: NDBuffer) -> Int:
    return 73


def compile[
    ArgTypes: TypeList[Trait=GPUPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


def main():
    var thing = compile[kernel]()
    var mtt = ManagedTileTensor()
    var db = DeviceBuffer[Int]()
    var ndb = NDBuffer()
    var result1 = thing.call(mtt, db, ndb)


# // -----

# Tests that variadic function thunks can handle memory types.


trait ZAnyMemType:
    pass


@fieldwise_init
struct ZMemType(ZAnyMemType, Movable where False):
    pass


@fieldwise_init
struct ZBool(ZAnyMemType, Movable where False):
    pass


def device_func(a: ZMemType, b: ZBool):
    pass


def enqueue[
    ArgTypes: TypeList[Trait=ZAnyMemType, ...],
    //,
    func: def(mut * args: * ArgTypes) thin -> None,
](*args: *ArgTypes):
    ...


# CHECK-LABEL: lit.fn @"do_thing
def do_thing(mut a: ZMemType, mut b: ZBool):
    # This checks that it:
    #  * Is calling enqueue
    #  * Is passing in the thunk (see that lit.generator's general shape)
    #  * Does a proper union of all the origins: imm { ... }
    #  * Also sends in the origin of the pack itself (the second origin parameter-value)
    # CHECK: lit.call {{.*}}@"enqueue{{[^"]*}}"[muttoimm *"anonymous*`2"]<:param_list<!ZAnyMemType_AnyType> [!ZMemType, !ZBool],
    # CHECK-SAME: :origin<false> {(mutcast mut *"a`"), (mutcast mut *"b`1")}
    # CHECK-SAME: {{.*}}: !lit.generator<[1]("args": !lit.ref<!lit.struct<#VariadicPack <:!Bool {:scalar<bool> false}, :origin<false> {(mutcast mut *"a`"), (mutcast mut *"b`1")}, :!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> {(mutcast mut *"a`"), (mutcast mut *"b`1")}>> {}, :meta<!AnyType> !ZAnyMemType_AnyType, :param_list<!ZAnyMemType_AnyType> [!ZMemType, !ZBool], :!Bool {:scalar<bool> false}, :!lit.struct<#TypeList <:meta<!AnyType> !ZAnyMemType_AnyType, :param_list<!ZAnyMemType_AnyType> [!ZMemType, !ZBool]>> {}>>, imm *[0,0]> imm_mem|pack_vararg) -> !kgen.none>
    enqueue[device_func](a, b)


# CHECK-LABEL:   lit.fn @"{{def[^"]*\|def[^"]*\|[^"]*}}"
# These check that it:
#  * Is taking in two origins; one for the VariadicPack itself, and a common origin for its elements
#  * It takes in a VariadicPack that references those origins correctly
# CHECK-SAME: [imm *"0_unnamed`"]
# CHECK-SAME: (%0[*""]: !lit.ref<!lit.struct<#VariadicPack <:!Bool {:scalar<bool> true}, :origin<true> _0, :!lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> _0>> _1, :meta<!AnyType> !ZAnyMemType_AnyType, :param_list<!ZAnyMemType_AnyType> [!ZMemType, !ZBool], :!Bool {:scalar<bool> false}, :!lit.struct<#TypeList <:meta<!AnyType> !ZAnyMemType_AnyType, :param_list<!ZAnyMemType_AnyType> [!ZMemType, !ZBool]>> {}>>, imm *"0_unnamed`"> imm_mem)

# // -----

# Tests that prepending clarifying parameters (TAPCPTTT) works correctly in the
# presence of existing parameters.
# Raw MLIR origin must be used for this test to test correctness, as the raw
# MLIR origin type has a strong requirement for the type of the attribute `mut`,
# whereas the Origin wrapper type does not.


comptime raw_origin[mut: __mlir_type.`!kgen.scalar<bool>`] = __mlir_type[`!lit.origin<`, mut, `>`]


trait Trait1:
    comptime RetType: AnyType

    def method1[
        mut: __mlir_type.`!kgen.scalar<bool>`, //, O: raw_origin[mut]
    ](ref[O] self) -> Self.RetType:
        ...


struct Struct1[x: Int](Trait1, TrivialRegisterPassable):
    comptime RetType = Int

    def method1[
        mut: __mlir_type.`!kgen.scalar<bool>`, //, O: raw_origin[mut]
    ](ref[O] self) -> Self.RetType:
        pass

# // -----

trait MyIndexable:
    def __getitem__(ref self, idx: Int) -> ref[origin_of(self).subtree] Int:
        ...

struct MyList(MyIndexable):
    var elements: List[Int]

    # More concrete origin so concrete clients get a tighter origin bound.
    def __getitem__(ref self, idx: Int) -> ref[self.elements[idx]] Int:
        return self.elements[idx]

# Check that the conversion thunk happens in terms of lit.ref.upcast
# CHECK-LABEL: lit.fn @"def[{{.*}}](ref[$1] self: {{.*}}MyList, idx: ::SIMD[{{.*}}]) thin -> ref
# CHECK-NEXT: %2 = lit.call tail[!lit.generator<defines_interior_origins
# CHECK-NEXT: %3 = lit.ref.upcast %2
# CHECK-NEXT: lit.return %3
