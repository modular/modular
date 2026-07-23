# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -o %t.mlir

# RUN: FileCheck %s --enable-var-scope --check-prefixes=S0 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S1 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S2 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S3 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S5 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S6 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S7 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S8 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S9 < %t.mlir
# COM: Verify ParamOperatorAttr and LITStructAttr matching: Pair(tag, 0) lowers
# COM: to #kgen.param.expr<apply, ...> containing #lit.struct constants, which
# COM: requires recursive matching through both composite attr types.
# S0-LABEL: lit.fn @"repro_struct_attr()"
# S0: lit.var.decl "my_fn" var : !lit.ref<!lit.struct<{{.*}} <:trait<@"def() -> Container[Pair(Int(2), Int(0))]"
# S0: lit.call @unified_closure_traits::@"struct_callee[::SIMD[::DType(int), ::SIMDLength(1)],def[tag: Int, //]() -> Container[Pair(tag, Int(0))]{1} & ::AnyType & ::ImplicitlyDeletable & ::Movable]($1){(eq $1.tag, $0)}"
# S0-SAME: <:!Int {:scalar<index> 2}



@fieldwise_init
struct Pair(RegisterPassable):
    var a: Int
    var b: Int


@fieldwise_init
struct Container[p: Pair](RegisterPassable):
    var value: Int


def struct_callee[
    tag: Int,
    F: def() -> Container[Pair(tag, 0)],
](closure: F):
    var result = closure()


def repro_struct_attr():
    var x = 10

    def my_fn() {read x} -> Container[Pair(2, 0)]:
        return Container[Pair(2, 0)](x)

    struct_callee[2, type_of(my_fn)](my_fn)

# COM: Verify SymbolConstantAttr matching: closure returning a type
# COM: parameterized by a function reference (exercises symbol recursion).
# S1-LABEL: lit.fn @"repro_symbol_attr()"
# S1-DAG: lit.var.decl "my_fn" var : !lit.ref<!lit.struct<{{.*}} <:trait<@"def() -> Dispatch[identity]"
# S1-DAG: lit.call @unified_closure_traits::@"symbol_callee[::SIMD[::DType(int), ::SIMDLength(1)],def() -> Dispatch[identity] & ::AnyType & ::ImplicitlyDeletable & ::Movable]($1)"{{.*}}<:!Int {:scalar<index> 1}



struct Dispatch[F: def(Int) thin -> Int]:
    var data: Int

    def __init__(out self, data: Int):
        self.data = data


def identity(x: Int) -> Int:
    return x


def symbol_callee[
    tag: Int,
    C: def() -> Dispatch[identity],
](closure: C):
    var result = closure()


def repro_symbol_attr():
    var x = 10

    def my_fn() {read x} -> Dispatch[identity]:
        return Dispatch[identity](x)

    symbol_callee[1, type_of(my_fn)](my_fn)

# COM: Ensure non-ref closure call operands are transformed/rebound in wrapper
# COM: __call__ before dispatching to impl witness call.
# S2-LABEL: lit.fn @"__call__[::SIMD[::DType(int), ::SIMDLength(1)],::SIMD[::DType(int), ::SIMDLength(1)]](unified_closure_traits::def[tag: Int, //, w: Int](val: Vec[tag, w]) -> Bool{1}_{{.*}}"
# S2: [[S2_REBIND:%.*]] = kgen.rebind %val : !lit.struct<#Vec <:!Int _tag
# S2-SAME: to !lit.struct<#Vec <:!Int #kgen.get_witness<:{{.*}} impl, "def[tag: Int, //, w: Int](val: Vec[tag, w]) -> Bool{1}", "tag">
# S2: lit.call[{{.*}}"val": !lit.struct<#Vec <:!Int #kgen.get_witness<:{{.*}} impl, "def[tag: Int, //, w: Int](val: Vec[tag, w]) -> Bool{1}", "tag">
# S2-SAME: ]{{.*}}(%{{.*}}, [[S2_REBIND]])



struct Width(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.index

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return self._mlir_value

    @implicit
    @always_inline
    def __init__[T: AnyType](out self, value: T):
        pass


struct Vec[tag: Int, size: Width](TrivialRegisterPassable):
    var _dummy: __mlir_type.i1


def repro_rebind_nonref_operand[
    tag: Int,
    F: def[w: Width](v: Vec[tag, w]) -> Bool,
](func: F):
    def body[w: Int](val: Vec[tag, w]) {read func} -> Bool:
        return func[w=w](val)

    _ = body

# S3: lit.struct.decl @"def() -> None_{{.*}}"
# S3: kgen.conformance @"std::builtin::{{.*}}::Copyable"
# S3-NEXT: kgen.witness "__init__(copy:$0)" : !lit.generator<[2](*, "copy":

struct s3_Foo(ImplicitlyCopyable, Movable):
    var x: Int
    var y: Int


def copyIt[X: Copyable](x: X):
    var copy = X.__init__(copy=x)


def thing(foo: s3_Foo):
    def thing() {var}:
        _ = foo

# COM: Overload resolution with a closure overload must not crash when the
# COM: non-closure argument's struct is not yet body-resolved.



@always_inline
def s4_dispatch[
    FuncType: TrivialRegisterPassable & def() -> None, //
](func: FuncType):
    pass


@always_inline
def s4_dispatch[T: AnyType](val: T):
    pass


def test(x: s4_Foo):
    s4_dispatch(x)


struct s4_Foo:
    var x: Int

# COM: Verify generic map where the actual closure returns in-register but the
# COM: trait signature expects a memory-only ByRefResult slot.
# S5-DAG: [[S5_INT:!Int.*]] = !lit.struct<#SIMD <{{.*}}>>
# S5-DAG: kgen.conformance @"def{{.*}}(x: T) -> U{2}" {
# S5-DAG:   kgen.witness "__call__{{.*}}" : !lit.generator
# S5-DAG:   kgen.witness "T" : {{.*}} = [[S5_INT]]
# S5-DAG:   kgen.witness "U" : {{.*}} = [[S5_INT]]




comptime CollectionElement = ImplicitlyDeletable & ImplicitlyCopyable


def s5_foo(x: Int):
    def map[
        T: CollectionElement,
        U: CollectionElement,
        func: def(x: T) -> U,
    ](item: T, closure: func) -> U:
        return closure(item)

    def double(x: Int) {mut} -> Int:
        return x * 2

    _ = map[Int, Int, type_of(double)](x, double)

# COM: Verify names match cache keys to avoid collisions.
# S6-DAG: lit.trait.decl @"def[U: DoB, //](y: U) -> None{1}"
# S6-DAG: lit.trait.decl @"def[T: DoA](y: T) -> None"
# S6-DAG: lit.trait.decl @"def[T: DoA, //](y: T) -> None{1}"



trait DoA:
    def doA(self):
        ...


trait DoB:
    def doB(self):
        ...


def s6_foo[T: DoA](x: T):
    def closure(y: T) {read x}:
        _ = x

    def closure2[T: DoA](y: T) {var}:
        pass


def bar[U: DoB](x: U):
    def closure(y: U) {var}:
        pass

# COM: Verify that a register_passable closure capturing a generic
# COM: register_passable closure and a concrete register_passable struct gets
# COM: convention register_passable (not trivial)
# S7-DAG: lit.struct.decl @"def(y: Int) -> Int_{{.*}}"{{.*}} register_passable attributes




struct NonTrivialPayload(ImplicitlyCopyable, RegisterPassable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def s7_call_inner[
    F: ImplicitlyCopyable & RegisterPassable & def(Int) -> Int
](f: F, x: Int) -> Int:
    var payload = NonTrivialPayload(1)

    def outer(y: Int) {var f, var payload} -> Int:
        return f(y) + payload.value

    return outer(x)

# COM: Verify that a register_passable closure capturing a trivially
# COM: register_passable callback and a trivial struct gets convention
# COM: register_passable_trivial.
# S8-DAG: lit.struct.decl @"def(y: Int) -> Int_{{.*}}"{{.*}} register_passable_trivial attributes




struct TrivialPayload(TrivialRegisterPassable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def s8_call_inner[
    F: TrivialRegisterPassable & def(Int) -> Int
](f: F, x: Int) -> Int:
    var payload = TrivialPayload(1)

    def outer(y: Int) {var f, var payload} -> Int:
        return f(y) + payload.value

    return outer(x)

# COM: Verify lazy conformance fires for a parametric closure trait whose
# COM: argument type is a (`param_list.get`).

# S9: kgen.conformance @"def[idx: Int](var elt: _[idx]) -> None{1}" {
# S9:   kgen.witness "__call__{{.*}} capturing -> !kgen.none>
# S9:   kgen.witness "element_types.values`" : param_list<{{.*}}> = [!String, !Int]

struct s9_MiniTuple[*element_types: Movable & ImplicitlyDeletable](Movable):
    comptime _mlir_type = __mlir_type[
        `!kgen.struct<:`,
        type_of(Self.element_types.values),
        Self.element_types.values,
        ` isParamPack>`,
    ]

    var _mlir_value: Self._mlir_type

    @always_inline("nodebug")
    def __getitem_param__[
        idx: Int
    ](ref self) -> ref [self] Self.element_types[idx]:
        var storage_kgen_ptr = UnsafePointer(
            to=self._mlir_value
        )._get_kgen_pointer()
        var elt_kgen_ptr = __mlir_op.`kgen.struct.gep`[
            index = idx.__mlir_index__(),
            _type = UnsafePointer[
                Self.element_types[idx], origin_of(self)
            ]._mlir_type,
        ](storage_kgen_ptr)
        return UnsafePointer[_, origin_of(self)](elt_kgen_ptr)[]

    @always_inline("nodebug")
    def consume_elements[
        EltHandler: def[idx: Int](var elt: Self.element_types[idx])
    ](deinit self, elt_handler: EltHandler, /):
        var ptr = UnsafePointer(to=self[0])
        elt_handler[0](__get_address_as_owned_value(ptr._get_kgen_pointer()))


def s9(var t: s9_MiniTuple[String, Int]):
    def handler[idx: Int](var elt: t.element_types[idx]) {var}:
        _ = elt^

    t^.consume_elements(handler)
