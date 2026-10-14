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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -o %t.mlir

# RUN: FileCheck %s --enable-var-scope --check-prefixes=S0 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S2 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S3 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S4 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S5 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S6 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S7 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S8 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S9 < %t.mlir
# COM: "U" cannot be called "T" until MOCO-4028 is fixed
# COM: The captured parameter becomes a parameter of the storage struct, and
# COM: the trait instance refers to it directly instead of through an alias on
# COM: a per-signature trait.
# S0: lit.struct.decl @"closure$makeIt{{.*}}::parametric::__storage"<U: !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable, {{.*}}>
# S0: kgen.conformance @"##__mojo_closure__##"{{.*}}:type #kgen.quote<!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable U>>
# COM: No bridging wrapper around storage.
# S0-NOT: lit.struct.decl @"extension$


def makeIt[U: TrivialRegisterPassable](a: U):
    def parametric() {var a} -> U:
        return a


def conditionallyDevicePassable(x: Int):
    def device_passable() {var} -> Int:
        return x


# COM: Ensure external parameter references are carried inside the closure
# COM: trait instance's quoted signature (they used to become alias decls on a
# COM: per-signature trait).
# S2-DAG: impl: !lit.ref<:trait<@"##__mojo_closure__##"{{.*}}#kgen.quote<!lit.ref<:!AnyType_DoIt T, imm *[0,1]>>
# S2-DAG: impl: !lit.ref<:trait<@"##__mojo_closure__##"{{.*}}#kgen.quote<!lit.ref<:!AnyType_DoIt TT, imm *[0,1]>>


trait DoIt:
    def thing(self):
        ...


struct House[T: DoIt](Movable where False):
    def aMethod[C: def(x: Self.T)](self, impl: C):
        pass


def useIt[TT: DoIt, C: def(x: TT)](impl: C):
    pass


# S3-DAG: kgen.conformance {{.*}}@RegisterPassable {


def takesRegisterPassable[T: RegisterPassable](impl: T):
    pass


def addTrivialRegisterPassable(x: Int):
    def closure() {var} -> Int:
        return x

    takesRegisterPassable(closure)


# COM: Verify top-level function symbols get conformance for count's closure
# COM: trait.
# S4: lit.struct.decl @"inflated$def[::SIMD[DType.int, 1]](vec: unified_closure_rebind::s4_ToySIMD{{.*}}|{{[0-9a-f]+}}"
# S4: kgen.conformance @"##__mojo_closure__##"{{.*}}ToySIMD{{.*}}{:scalar<index> 1}{{.*}}ToyMask{{.*}}{:scalar<index> 1}
# S4-NEXT: kgen.witness "__call__" : !lit.generator


@fieldwise_init
struct s4_ToyBool(Movable where False):
    var value: Int


@fieldwise_init
struct s4_ToyMask[dtype_tag: Int, w: Int](Movable where False):
    var value: Int


struct s4_ToySIMD[dtype_tag: Int, w: Int](Movable where False):
    pass


struct s4_ToyScalar[dtype_tag: Int](Movable where False):
    pass


@fieldwise_init
struct s4_MiniSpan[dtype_tag: Int](Movable where False):
    var value: Int

    def count[
        F: def[w: Int](vec: s4_ToySIMD[Self.dtype_tag, w]) -> s4_ToyMask[
            Self.dtype_tag, w
        ]
    ](self, func: F) -> Int:
        return 0


def is_vec_a[w: Int](vec: s4_ToySIMD[1, w]) -> s4_ToyMask[1, w]:
    _ = vec
    return s4_ToyMask[1, w](0)


def repro_top_level():
    var s = s4_MiniSpan[1](0)
    _ = s.count(is_vec_a)


# COM: Verify nested captured closures get conformance for count's
# COM: closure trait on the storage struct (no parametric wrapper).
# S5: lit.struct.decl @"{{.*}}is_vec_a_capturing::__storage"
# S5: kgen.conformance @"##__mojo_closure__##"{{.*}}ToySIMD{{.*}}{:scalar<index> 1}{{.*}}ToyMask{{.*}}{:scalar<index> 1}
# S5-NEXT: kgen.witness "__call__" : !lit.generator


@fieldwise_init
struct s5_ToyBool(Movable where False):
    var value: Int


@fieldwise_init
struct s5_ToyMask[dtype_tag: Int, u: Int](Movable where False):
    var value: Int


struct s5_ToySIMD[dtype_tag: Int, u: Int](Movable where False):
    pass


struct s5_ToyScalar[dtype_tag: Int](Movable where False):
    pass


@fieldwise_init
struct s5_MiniSpan[dtype_tag: Int](Movable where False):
    var value: Int

    def count[
        F: def[u: Int](vec: s5_ToySIMD[Self.dtype_tag, u]) -> s5_ToyMask[
            Self.dtype_tag, u
        ]
    ](self, func: F) -> Int:
        return 0


def repro_capturing(mem: String):
    var capture = 0

    def is_vec_a_capturing[
        u: Int
    ](vec: s5_ToySIMD[1, u]) {var capture, var mem} -> s5_ToyMask[1, u]:
        _ = vec
        _ = capture
        return s5_ToyMask[1, u](0)

    var s = s5_MiniSpan[1](0)
    _ = s.count(is_vec_a_capturing)


# COM: Verify nested type parameters constrained by a trait (not just Int
# COM: parameters) get conformance resolved from nested struct type arguments.
# S6: lit.struct.decl @"{{.*}}apply_concrete::__storage"
# S6: kgen.conformance @"##__mojo_closure__##"{{.*}}#Box <:!AnyType_ElemLike !ConcreteElem
# S6-NEXT: kgen.witness "__call__" : !lit.generator


trait ElemLike:
    pass


struct ConcreteElem(ElemLike, Movable where False):
    pass


@fieldwise_init
struct Box[E: ElemLike, n: Int](Movable where False):
    var value: Int


@fieldwise_init
struct Store[E: ElemLike](Movable where False):
    var value: Int

    def apply[
        F: def[n: Int](item: Box[Self.E, n]) -> Box[Self.E, n]
    ](self, func: F) -> Int:
        return 0


def repro_nested_type_param(mem: String):
    var capture = 0

    def apply_concrete[
        n: Int
    ](item: Box[ConcreteElem, n]) {var capture, var mem} -> Box[
        ConcreteElem, n
    ]:
        _ = item
        _ = capture
        return Box[ConcreteElem, n](0)

    var s = Store[ConcreteElem](0)
    _ = s.apply(apply_concrete)


# COM: Verify that custom types (the result type !kgen.none in this case) are compared using equality
# S7: lit.struct.decl @"{{.*}}my_func::__storage"
# S7: kgen.conformance @"##__mojo_closure__##"<:param_list<type> [#kgen.quote<!Int>, #kgen.quote<!Int>, #kgen.quote<!Int>], :param_list<type> [], :type #kgen.quote<none>
# S7-NEXT:   kgen.witness "__call__" : !lit.generator


def print(x: Int):
    pass


def s7_callee[
    func: def[width: Int, rank: Int, alignment: Int = 1]() -> None,
    //,
    simd_width: Int,
](shape: Int, ctx: Int, closure: func):
    closure[simd_width, 2]()


def main() raises:
    var x = 42
    var mem: String = "hello"

    @always_inline
    def my_func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ]() {imm x, var mem}:
        print(x)

    s7_callee[simd_width=4](10, 11, my_func)


# COM: Verify the result is properly rebound in the struct wrapper when a closure
# COM: lazily conforms to a trait whose return type contains an alias parameter.
# S8: lit.struct.decl @"inflated$def[::SIMD[DType.int, 1]]() thin -> unified_closure_rebind::V[::SIMD[DType.int, 1](42), $0]|{{[0-9a-f]+}}"
# S8: lit.fn @"__call__[::SIMD[DType.int, 1]](inflated$def{{.*}} -> unified_closure_rebind::V{{.*}})"
# COM: `bind_params` is what re-binds the trait's `width` onto the promoted
# COM: function's own parameter, so the result needs no separate rebind.
# S8: lit.call tail[!lit.generator<() -> !lit.struct<#V <:!Int {:scalar<index> 42}, :!Int *"Closure_Syn#0">>>: bind_params(
# S8-NEXT: lit.return %{{.*}} : !lit.struct<#V <:!Int {:scalar<index> 42}, :!Int *"Closure_Syn#0">>
# S8: kgen.conformance @"##__mojo_closure__##"{{.*}}:type #kgen.quote<!lit.struct<#V <:!Int {:scalar<index> 42}, :!Int *(0,1)>>>
# S8-NEXT: kgen.witness "__call__" : !lit.generator


@fieldwise_init
struct V[dtype: Int, width: Int](RegisterPassable):
    var _v: Int


def s8_callee[
    dtype: Int,
    F: RegisterPassable & def[width: Int]() -> V[dtype, width],
](closure: F):
    var result = closure[4]()


def rebindResult():
    def my_closure[width: Int]() {} -> V[42, width]:
        return V[42, width](0)

    s8_callee[42](my_closure)


# COM: Verify ParamListAttr matching: closure returning Tuple with parameterized
# COM: elements requires recursive matching through #kgen.param_list param values.
# S9: lit.struct.decl @"{{.*}}my_map_fn::__storage"
# S9: kgen.conformance @"##__mojo_closure__##"{{.*}}#ToyIndex <:!Int {:scalar<index> 2}
# S9-NEXT:   kgen.witness "__call__" : !lit.generator


struct ToyIndex[size: Int](RegisterPassable):
    var _v: Int

    def __init__(out self):
        self._v = 0


def variadic_callee[
    rank: Int,
    map_fn: def(ToyIndex[rank]) -> Tuple[
        ToyIndex[rank],
        ToyIndex[rank],
    ],
](closure: map_fn):
    var point = ToyIndex[rank]()
    var result = closure(point)


def repro_variadic_attr():
    var x = 10
    var mem: String = "hello"

    def my_map_fn(
        point: ToyIndex[2],
    ) {imm x, var mem} -> Tuple[ToyIndex[2], ToyIndex[2]]:
        return ToyIndex[2](), ToyIndex[2]()

    variadic_callee[2, type_of(my_map_fn)](my_map_fn)
