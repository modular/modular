# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt --kgen-print-inline-type-values | FileCheck %s


# CHECK-LABEL: lit.trait.decl @Trait
# CHECK-SAME: <?, [[T:.*]]: !Trait>
trait Trait:
    # CHECK: lit.fn @"f0{{.*}}(%self: !lit.ref<:!Trait [[T]], imm {{.*}}> read_mem) -> !kgen.none
    # CHECK-NEXT: kgen.unreachable
    fn f0(self):
        ...

    # CHECK: lit.fn @"f1{{.*}}(%self: !lit.ref<{{.*}}> mut) -> !kgen.none
    # CHECK-NEXT: kgen.unreachable
    fn f1(mut self):
        ...

    # CHECK: lit.fn @"f2{{.*}}(%self: !lit.ref<{{.*}}> mut) -> !kgen.none attributes {defaultedTraitFn,
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.return %none : !kgen.none
    # CHECK-NEXT: lit.end_fn
    fn f2(mut self):
        pass

    # CHECK: lit.fn @"f3{{.*}}(%self: !lit.ref<{{.*}}> read_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, mut *"__result__`2x2"> byref_result) throws -> i1
    # CHECK-NEXT: kgen.unreachable
    def f3(self):
        ...

    # CHECK: lit.fn @"f4{{.*}}(%self: !lit.ref<{{.*}}> read_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, mut *"__result__`2x2"> byref_result) throws -> i1
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.ref.store %none, %__result__ : <none, mut *"__result__`2x2">
    # CHECK-NEXT: %0 = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return %0 : i1
    # CHECK-NEXT: lit.end_fn
    def f4(self):
        pass

    # CHECK: lit.fn @"f5{{.*}}(%self: !lit.ref<{{.*}}> mut, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, {{.*}}> byref_result) throws -> i1
    # CHECK-NEXT: kgen.unreachable
    def f5(mut self):
        ...

    # CHECK: lit.fn @"f6{{.*}}(%self: !lit.ref<{{.*}}> mut, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, {{.*}}> byref_result) throws -> i1
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.ref.store %none, %__result__ : <none, mut *"__result__`2x2">
    # CHECK-NEXT: %0 = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return %0 : i1
    # CHECK-NEXT: lit.end_fn
    def f6(mut self):
        pass

    fn overloaded(self):
        ...

    fn overloaded(self, x: Int):
        ...

    fn overloaded(self, x: string):
        ...

    # CHECK-LABEL: lit.fn @"parametric{{.*}}<x: !Int>
    fn parametric[x: Int](self):
        ...


# CHECK-LABEL: lit.trait.decl @EmptyTrait
trait EmptyTrait:
    pass


# CHECK-LABEL: lit.trait.decl @Trait1
# CHECK-SAME: <?, [[T:.*]]: !Trait1>
trait Trait1:
    # CHECK: lit.fn @"f{{.*}}(%self: !lit.ref<{{.*}}> read_mem, ?, %__result__: !lit.ref<:!Trait1 [[T]], mut {{.*}}> byref_result) -> !kgen.none
    fn f(self) -> Self:
        ...


trait Trait2:
    fn f(self) -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @StructWithTraits({{.*}}Trait1_Trait2)
struct StructWithTraits(Trait1, Trait2):
    # CHECK: lit.fn @"f{{.*}}(%self: !lit.ref<!StructWithTraits, imm {{.*}}> read_mem, ?, %{{.*}}: !lit.ref<!StructWithTraits, mut {{.*}}> byref_result) -> !kgen.none
    fn f(self) -> Self:
        ...


# CHECK-LABEL: lit.trait.decl @CFMTrait
trait CFMTrait:
    # CHECK: lit.fn @"f1{{.*}}(%self: !lit.ref<{{.*}}> read_mem) -> !kgen.none
    fn f1(self):
        pass

    # CHECK: lit.fn @"f2()"() -> !kgen.none
    @staticmethod
    fn f2():
        pass


# CHECK-LABEL: lit.struct.decl @CFMStruct({{.*}}CFMTrait)
struct CFMStruct(CFMTrait):
    # CHECK: lit.fn @"f1({{.*}})"[{{.*}}](%self: !lit.ref<!CFMStruct, imm {{.*}}> read_mem) -> !kgen.none
    fn f1(self):
        pass

    # CHECK: lit.fn @"f2()"() -> !kgen.none
    @staticmethod
    fn f2():
        pass


# Test for struct with parameters and function with parameters.
# CHECK-LABEL: lit.trait.decl @CFMTraitParams
trait CFMTraitParams:
    # CHECK: lit.fn @"f1{{.*}}"<x: !CFMTraitParams>[{{.*}}](
    fn f1[x: CFMTraitParams](self):
        pass


# CHECK-LABEL: lit.struct.decl @CFMStructParams
struct CFMStructParams[t1: AnyTrivialRegType, t2: AnyTrivialRegType](
    CFMTraitParams
):
    # CHECK: lit.fn @"f1{{.*}}"<x: !CFMTraitParams>[{{.*}}](%self: !lit.ref<!lit.struct<#CFMStructParams <:type t1, :type t2>>{{.*}}> read_mem)
    fn f1[x: CFMTraitParams](self):
        pass


# ===----------------------------------------------------------------------=== #
# Call Emission
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"generic_trait_fn{{.*}}<T: !Trait>
# CHECK-SAME: %x: !lit.ref<:!Trait T, imm {{.*}}> read_mem
fn generic_trait_fn[T: Trait](x: T):
    # CHECK: lit.call[!lit.generator<[1]("self": {{.*}} read_mem) -> !kgen.none>:
    # CHECK-SAME: #kgen.get_witness<:!Trait T, "traits::Trait", "f0{{.*}}">]{{.*}}(%x)
    x.f0()

    # CHECK: lit.call[!lit.generator<[1]("self": {{[^)]*}}) -> !kgen.none>:
    # CHECK-SAME: #kgen.get_witness<:!Trait T, "traits::Trait", "overloaded{{.*}}">]{{.*}}(%x)
    x.overloaded()
    # CHECK: lit.call[!lit.generator<[1]("self": {{.*}}, "x": !Int)
    # CHECK-SAME: #kgen.get_witness<:!Trait T, "traits::Trait", "overloaded{{.*}}">]{{.*}}(%x, %{{.*}})
    x.overloaded(1)
    # CHECK: lit.call[!lit.generator<[1]("self": {{.*}}, "x": !kgen.string)
    # CHECK-SAME: #kgen.get_witness<:!Trait T, "traits::Trait", "overloaded{{.*}}">]{{.*}}(%x, %{{.*}})
    x.overloaded(__mlir_attr.`"trait" : !kgen.string`)

    # CHECK: lit.call[!lit.generator<[1]("self": {{[^)]*}} read_mem)
    # CHECK-SAME: bind_params(:!lit.generator<<"x": !Int>[1](
    # CHECK-SAME: #kgen.get_witness<:!Trait T, "traits::Trait", "parametric{{.*}}">, {{.*}}1{{.*}})
    x.parametric[1]()


# CHECK-LABEL: lit.fn @"existential_arg
# CHECK-SAME: (%x: !lit.ref<!Trait, imm {{.*}}>
fn existential_arg(x: Trait):
    pass


trait SimpleTrait(ImplicitlyCopyable):
    fn method(self, y: Int):
        ...

    fn param_method[x: Int](self):
        ...


struct TraitStruct(SimpleTrait):
    fn method(self, y: Int):
        pass

    fn param_method[x: Int](self):
        pass


struct ParametricTraitStruct[z: Int](SimpleTrait):
    fn method(self, y: Int):
        pass

    fn param_method[x: Int](self):
        pass


fn take_simple_trait[T: SimpleTrait]():
    pass


fn infer_trait[T: SimpleTrait](value: T):
    pass


# CHECK-LABEL: lit.fn @"test_metatype_to_trait
fn test_metatype_to_trait():
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait !TraitStruct
    take_simple_trait[TraitStruct]()
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait {{.*}}@ParametricTraitStruct<:!Int {2}>
    take_simple_trait[ParametricTraitStruct[2]]()


# CHECK-LABEL: lit.fn @"test_infer_trait
fn test_infer_trait(
    a: TraitStruct, b: ParametricTraitStruct[2]
):
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait !TraitStruct
    infer_trait(a)
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait {{.*}}@ParametricTraitStruct<:!Int {2}>
    infer_trait(b)


trait StaticMethodTrait:
    @staticmethod
    fn foobar():
        pass


struct StaticMethodStruct(StaticMethodTrait, ImplicitlyCopyable):
    @staticmethod
    fn foobar():
        pass

    fn __copyinit__(out self, existing: Self):
        pass


# CHECK-LABEL: lit.fn @"trait_static_method{{.*}}<T: !StaticMethodTrait
fn trait_static_method[T: StaticMethodTrait]():
    # CHECK: call[!lit.generator<() -> !kgen.none>: #kgen.get_witness<:!StaticMethodTrait T, "traits::StaticMethodTrait", "foobar{{.*}}">]()
    T.foobar()


# CHECK-LABEL: lit.fn @"copy_me
# CHECK-SAME: <T: !ImplicitlyCopyable
# CHECK-SAME: %value: !lit.ref<:!ImplicitlyCopyable T, imm {{.*}}> read_mem, ?,
# CHECK-SAME: %__result__: !lit.ref<:!ImplicitlyCopyable T, mut {{.*}}> byref_result
fn copy_me[T: ImplicitlyCopyable](value: T) -> T:
    # CHECK-NEXT: call[!lit.generator<[2]("existing": {{.*}}T, {{.*}}> read_mem, |, ?, "self": {{.*}}T, {{.*}}> byref_result) -> !kgen.none>:
    # CHECK-SAME: #kgen.get_witness<:!ImplicitlyCopyable T, "stdlib::builtin::stubs::Copyable", "__copyinit__{{.*}}">]{{.*}}(%value, %__result__)
    return value


# CHECK-LABEL: lit.fn @"move_me
# CHECK-SAME: <T: !Movable
# CHECK-SAME: :!Movable T, {{.*}}> owned_in_mem
# CHECK-SAME: :!Movable T, {{.*}}> byref_result
fn move_me[T: Movable](var value: T) -> T:
    # CHECK-NEXT: lit.ownership.use %value
    # CHECK-NEXT: call[{{.*}}#kgen.get_witness<:!Movable T, "stdlib::builtin::stubs::Movable", "__moveinit__{{.*}}">]{{.*}}(%value, %__result__)
    return value^


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    @implicit
    fn __init__(out self, x: Int):
        ...

    fn __copyinit__(out self, existing: Self):
        ...

    @staticmethod
    fn may_throw() raises -> Self:
        ...

    fn throwing_method(self) raises:
        ...


# ===----------------------------------------------------------------------=== #
# Calling Convention / Register Passable
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @RegTraitType
@register_passable
struct RegTraitType(TraitForReg):
    # CHECK-LABEL: lit.fn @"__init__
    # CHECK-SAME: (%x: !Int) -> !RegTraitType
    @implicit
    fn __init__(out self, x: Int):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn may_throw() raises -> Self:
        pass

    fn throwing_method(self) raises:
        pass


# CHECK-LABEL: lit.fn @"raising_method
fn raising_method[T: TraitForReg](x: T) raises:
    # CHECK: lit.call[{{.*}}: #kgen.get_witness<:!TraitForReg T, "traits::TraitForReg", "may_throw{{.*}}">][{{.*}}](%__error__, %__call_result_tmp__
    _ = T.may_throw()
    # CHECK: lit.call[{{.*}}: #kgen.get_witness<:!TraitForReg T, "traits::TraitForReg", "throwing_method{{.*}}">][{{.*}}](%{{.*}}, %__error__, %__call_result_tmp__
    x.throwing_method()


trait CrazyTrait:
    pass

    fn foo[b: Int](self, c: Int) -> Self:
        ...


trait ChangedResultTypeTrait:
    @staticmethod
    fn result_type() -> Self:
        ...


# COM: The calling convention rewrite results in a decl with two "overloads" that
# COM: differ only in result type. Ensure that the thunk gets selected.
# CHECK-LABEL: lit.struct.decl @ChangedResultTypeStruct
@register_passable
struct ChangedResultTypeStruct(ChangedResultTypeTrait):
    # CHECK-LABEL: lit.fn @"result_type()"() -> !ChangedResultTypeStruct
    @staticmethod
    fn result_type() -> Self:
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}ChangedResultTypeTrait
    # CHECK-NEXT: kgen.witness "result_type{{.*}}" : !lit.generator<{{.*}}"__result__": !lit.ref<!ChangedResultTypeStruct, {{.*}}> byref_result) -> !kgen.none>{{.*}}fn() -> traits::ChangedResultTypeStruct

# CHECK-LABEL: lit.fn @"convert_result_type
fn convert_result_type():
    @parameter
    fn convert_result_type[T: ChangedResultTypeTrait]():
        pass

    # CHECK: call{{.*}}!ChangedResultTypeStruct
    convert_result_type[ChangedResultTypeStruct]()


trait SimpleTraitMethod:
    fn foo(self):
        ...


@register_passable
struct VariadicTrait[*I: Int](SimpleTraitMethod):
    fn foo(self):
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}SimpleTraitMethod
    # CHECK-NEXT: kgen.witness "foo{{.*}}" : !lit.generator<[1]("self": {{.*}}<:variadic<!Int> I>{{.*}} read_mem) -> !kgen.none> = {{.*}}@"foo{{.*}}"<:variadic<!Int> I>

# CHECK-LABEL: lit.fn @"test_bind_variadic
fn test_bind_variadic():
    @parameter
    fn bind_trait[T: SimpleTraitMethod]():
        pass

    # CHECK: call{{.*}}@VariadicTrait<:variadic<!Int> []>
    bind_trait[VariadicTrait[]]()


trait ThunkAmbiguity:
    fn mismatched_arg(self):
        ...

    @staticmethod
    fn mismatched_ret() -> Self:
        ...

    fn __init__(out self):
        ...


@register_passable
struct ThunkAmbiguityRP(ThunkAmbiguity):
    fn mismatched_arg(self):
        pass

    @staticmethod
    fn mismatched_ret() -> Self:
        pass

    fn __init__(out self):
        pass


# COM: Make sure that the generated thunks aren't select over the methods.


# CHECK-LABEL: lit.fn @"ambiguous_thunk
fn ambiguous_thunk(x: ThunkAmbiguityRP):
    # CHECK-NOT: `thunk
    x.mismatched_arg()
    _ = ThunkAmbiguityRP.mismatched_ret()
    _ = ThunkAmbiguityRP()
    # CHECK-LABEL: lit.end_fn


trait OwnedArguments:
    fn take(var self, var x: RegTraitType):
        ...


# CHECK-LABEL: lit.struct.decl @NoDtor
@register_passable
struct NoDtor(OwnedArguments, DefaultConstructible):
    fn take(var self, var x: RegTraitType):
        pass

    fn __init__(out self):
        pass

    fn method(self):
        pass


trait DefaultConstructible:
    fn __init__(out self):
        ...


fn default_construct[T: DefaultConstructible]() -> T:
    return T()


# CHECK-LABEL: lit.fn @"generic_fn_return_type
fn generic_fn_return_type():
    # CHECK: lit.var.decl "c" var : !lit.ref<!NoDtor,
    # CHECK-NEXT: call {{.*}}default_construct{{.*}}<:!DefaultConstructible !NoDtor>(%c)
    var c = default_construct[NoDtor]()
    # CHECK: call {{.*}}@NoDtor::@"method
    c.method()


trait SimpleTraitA:
    fn method(self):
        ...


trait SimpleTraitB:
    fn method(self):
        ...


# CHECK-LABEL: lit.struct.decl @TwoThunks
# CHECK-SAME: (!AnyType_Movable_UnknownDestructibility_SimpleTraitA_SimpleTraitB)
@register_passable
struct TwoThunks(SimpleTraitA, SimpleTraitB):
    # CHECK: lit.fn @"method({{.*}}TwoThunks)"
    fn method(self):
        pass


# https://linear.app/modularml/issue/MOCO-335/[bug]-register-passable-generates-phantom-trait-bound-overload
# CHECK-LABEL: lit.fn @"regpassable_reference
fn regpassable_reference():
    # CHECK-NEXT: @TwoThunks::@"method
    comptime f = TwoThunks.method


trait RequiredType:
    comptime T: AnyType

    @staticmethod
    fn use_it(arg: Self.T) -> Self.T:
        ...


struct RegPassableRequiredType(RequiredType):
    comptime T = Int

    @staticmethod
    fn use_it(arg: Int) -> Int:
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}RequiredType
    # CHECK: kgen.witness "use_it{{.*}}" : {{.*}}fn(::Int) -> ::Int


# CHECK-LABEL: lit.fn @"bind_regpassable_required_type
fn bind_regpassable_required_type():
    # CHECK-NEXT: : !RequiredType = <!RegPassableRequiredType>
    comptime T: RequiredType = RegPassableRequiredType


# ===----------------------------------------------------------------------=== #
# Special Functions
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @RegTrivialSpecial
@register_passable("trivial")
struct RegTrivialSpecial(AnyType, ImplicitlyCopyable, Movable):
    pass
    # CHECK: lit.fn @"__del__
    # CHECK: lit.fn @"__moveinit__
    # CHECK: lit.fn @"__copyinit__


# CHECK-LABEL: lit.struct.decl @RegSpecial
@register_passable
struct RegSpecial(AnyType, ImplicitlyCopyable, Movable):
    fn __copyinit__(out self, existing: Self):
        pass

    # CHECK: lit.fn @"__del__
    # CHECK: lit.fn @"__moveinit__


# CHECK-LABEL: lit.struct.decl @MemoryOnlySpecial
struct MemoryOnlySpecial(AnyType, ImplicitlyCopyable, Movable):
    pass
    # CHECK: lit.fn @"__del__
    # CHECK-SAME: [{{.*}} deinit_mem, |) -> !kgen.none
    # CHECK: return %none


fn copy[T: ImplicitlyCopyable](x: T):
    pass


fn move[T: Movable](x: T):
    pass


fn destroy[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test_special_fn_traits
fn test_special_fn_traits(
    mut x: RegTrivialSpecial, mut y: RegSpecial, mut z: MemoryOnlySpecial
):
    # COM: Just check that the implicit conversion succeeds.
    # CHECK-COUNT-9: lit.call
    copy(x)
    move(x)
    destroy(x)
    copy(y)
    move(y)
    destroy(y)
    copy(z)
    move(z)
    destroy(z)


# ===----------------------------------------------------------------------=== #
# Inheritance
# ===----------------------------------------------------------------------=== #


trait ParentTraitSameSig:
    fn foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @ChildTraitSameSig
trait ChildTraitSameSig(ParentTraitSameSig):
    # CHECK-NEXT: lit.fn @"foo
    # CHECK-NEXT: kgen.unreachable
    fn foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @GreatGrandFather
# CHECK-SAME: (!AnyType_UnknownDestructibility_GreatGrandFather)
trait GreatGrandFather:
    # CHECK: lit.fn @"foo
    fn foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @GrandFather
# CHECK-SAME: GreatGrandFather)
# CHECK-SAME: immediateParents = #M<symbols[@traits::@GreatGrandFather]>
trait GrandFather(GreatGrandFather):
    # CHECK: lit.fn @"bar
    fn bar(self):
        ...

    # CHECK: lit.fn @"foo


# CHECK-LABEL: lit.trait.decl @Father
# CHECK-SAME: GrandFather_GreatGrandFather)
# CHECK-SAME: immediateParents = #M<symbols[@traits::@GrandFather]>
trait Father(GrandFather):
    # CHECK: lit.fn @"baz
    fn baz(self):
        ...

    # CHECK: lit.fn @"bar
    # CHECK: lit.fn @"foo


# CHECK-LABEL: lit.trait.decl @UnevenDiamond
# CHECK-SAME: Father_GrandFather_GreatGrandFather_UnevenDiamond)
# CHECK-SAME: immediateParents = #M<symbols[@traits::@Father]>
trait UnevenDiamond(GreatGrandFather, Father):
    ...


# CHECK-LABEL: lit.struct.decl @TraitInheritance
# CHECK-SAME: Father_GrandFather_GreatGrandFather)
struct TraitInheritance(Father):
    fn foo(self):
        pass

    fn bar(self):
        pass

    fn baz(self):
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}Father
    # CHECK-NEXT: kgen.witness "baz{{.*}}"

    # CHECK-LABEL: kgen.conformance @{{.*}}GrandFather
    # CHECK-NEXT: kgen.witness "bar{{.*}}"

    # CHECK-LABEL: kgen.conformance @{{.*}}GreatGrandFather
    # CHECK-NEXT: kgen.witness "foo{{.*}}"


# CHECK-LABEL: lit.fn @"test_trait_inheritance
fn test_trait_inheritance():
    @parameter
    fn take_great_grand_father[T: GreatGrandFather]():
        pass

    @parameter
    fn take_grand_father[T: GrandFather]():
        pass

    @parameter
    fn take_father[T: Father]():
        pass

    # CHECK: call{{.*}}!TraitInheritance
    take_great_grand_father[TraitInheritance]()
    # CHECK: call{{.*}}!TraitInheritance
    take_grand_father[TraitInheritance]()
    # CHECK: call{{.*}}!TraitInheritance
    take_father[TraitInheritance]()


fn infer_grand_father[T: GrandFather](x: T):
    pass


# CHECK-LABEL: lit.fn @"pass_up_trait
# CHECK-SAME: <T: !Father>
fn pass_up_trait[T: Father](x: T):
    # CHECK-NEXT: call {{.*}}infer_grand_father{{.*}}<:!GrandFather !kgen.param<:!Father T>>(%x)
    infer_grand_father(x)


# ===----------------------------------------------------------------------=== #
# Misc Bugs
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct MovableType[T: Movable]:
    pass


trait InCollection(Movable):
    pass


struct Collection[T: InCollection]:
    var x: MovableType[Self.T]


@register_passable("trivial")
# CHECK-LABEL: lit.struct.decl @Item
struct Item(InCollection):
    pass

    # CHECK-LABEL: kgen.conformance @{{.*}}Movable
    # CHECK-NEXT: kgen.witness "__moveinit__{{.*}}"


fn take_movable(x: MovableType[Item]):
    pass


# CHECK-LABEL: lit.fn @"converted_metatype_struct_element
fn converted_metatype_struct_element(x: Collection[Item]):
    # CHECK: call {{.*}}take_movable
    take_movable(x.x)


# CHECK-LABEL: lit.struct.decl @TraitMember
# CHECK-NEXT: destructor
struct TraitMember[T: Movable]:
    # CHECK: lit.fn @"__del__
    var value: Self.T


# COM: Misleading error about thunk functions when: (issue mojo-#1402)
#      the test has
#      - a struct conforms to a trait, e.g. Movable
#      - the struct has a field of another type with parameter as itself, e.g MyPointer[Self]
#      - the field struct type's parameter should conform to Movable


# CHECK-LABEL: lit.struct.decl @MyPointer
@fieldwise_init
struct MyPointer[T: AnyType](ImplicitlyCopyable, Movable):
    pass
    # CHECK: lit.fn @"__del__
    # CHECK: lit.fn @"__init__


# CHECK-LABEL: lit.struct.decl @HasMyPointerSelf
struct HasMyPointerSelf(AnyType):
    # CHECK: lit.struct.field x : !lit.struct<#MyPointer <:!AnyType
    var x: MyPointer[Self]
    # CHECK: lit.fn @"__del__

    fn __moveinit__(out self, deinit existing: Self, /):
        pass


# Parser crash
# https://github.com/modularml/modular/issues/27897
# CHECK-LABEL: lit.fn @"check_trait_conversion_bymem_result_alias_crash
fn retMemory[T: TraitForReg](value: T) -> MemoryOnlySpecial:
    pass


fn check_trait_conversion_bymem_result_alias_crash(
    x: RegTraitType,
) -> MemoryOnlySpecial:
    return retMemory(x)


# Calling functions with implicit origins needs to cooperate.
fn test[a: ABC]():
    _ = ABCOptionalParamInt[ABCDim(a)]()


trait SomeTrait:
    pass


struct ABC(SomeTrait):
    fn __init__(out self):
        pass


@register_passable("trivial")
struct ABCOptionalParamInt[dim_parametric: ABCDim]:
    fn __init__(out self):
        pass


struct ABCDim:
    fn __init__[type: SomeTrait](out self, value: type):
        pass


trait TraitParameterized:
    fn foo[T: SomeTrait](self):
        ...


struct ConcreteType(TraitParameterized):
    fn foo[T: SomeTrait](self):
        pass


trait KeysBuilder:
    fn add[x: Int](mut self):
        ...


struct KeysContainer[end: Int](KeysBuilder):
    fn add[x: Int](mut self):
        pass


# CHECK-LABEL: lit.fn @"param_trait
fn param_trait[T: SimpleTrait, value: T]():
    # CHECK-NEXT: apply({{.*}} #kgen.get_witness<:!SimpleTrait T, "traits::SimpleTrait", "method{{.*}}">{{.*}} store_to_mem(value), {{.*}}1{{.*}})
    comptime param = value.method(1)
    # CHECK-NEXT: [[VAR:%.*]] = lit.var.decl
    # CHECK-NEXT: [[VALUE:%.*]] = kgen.param.materialize
    # CHECK-NEXT: store [[VALUE]], [[VAR]]
    # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[VAR]]
    # CHECK: call[{{.*}}#kgen.get_witness<:!SimpleTrait T, "traits::SimpleTrait", "method{{.*}}">{{.*}}([[IMM]],
    value.method(2)


trait Makeable:
    @staticmethod
    fn make() -> Self:
        ...


@register_passable
struct MakeNamedResult(Makeable):
    @staticmethod
    fn make(out out: Self):
        pass



# CHECK-LABEL: lit.fn @"check_named_result_regpassable
fn check_named_result_regpassable():
    # CHECK-NEXET: @MakeNamedResult::@"make()"
    comptime T: Makeable = MakeNamedResult


# COM: Issue https://github.com/modularml/modular/issues/33939
# COM: Ensure parameter inference works between type value attributes.
trait OtherEmptyTrait(EmptyTrait):
    pass


struct Bar[T: EmptyTrait]:
    pass


struct Foo[T: EmptyTrait]:
    fn infer_sub_trait[OT: OtherEmptyTrait](mut self, existing: Bar[OT]):
        pass


# CHECK-LABEL: lit.fn @"test_infer_sub_trait
fn test_infer_sub_trait[T: OtherEmptyTrait](var foo: Foo[T], bar: Bar[T]):
    # CHECK: call {{.*}}@Foo::@"infer_sub_trait{{.*}}<:!EmptyTrait !kgen.param<:!OtherEmptyTrait T>, :!OtherEmptyTrait T>(%foo, %bar)
    var copy = foo.infer_sub_trait(bar)


# ===----------------------------------------------------------------------=== #
# AnyTrait subtyping
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"anytrait_assignment
fn anytrait_assignment():
    # CHECK-NEXT: !lit.anytrait<!AnyType_Movable> = <!AnyType_Movable>
    comptime t: type_of(AnyType & Movable) = AnyType&Movable


# CHECK-LABEL: lit.fn @"test_anytrait_subtyping
# CHECK-SAME: <ty: !lit.anytrait<!UnknownDestructibility>>
fn test_anytrait_subtyping[ty: type_of(UnknownDestructibility)]():
    # Call !lit.anytrait subtyping.
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:!lit.anytrait<!UnknownDestructibility> !UnknownDestructibility>()
    test_anytrait_subtyping[UnknownDestructibility]()
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:!lit.anytrait<!UnknownDestructibility> !SimpleTrait>()
    test_anytrait_subtyping[SimpleTrait]()


# CHECK-LABEL: lit.fn @"take_many_things_of_specified_trait
# CHECK-SAME: <element_type: !lit.anytrait<!UnknownDestructibility>,
# CHECK-SAME: element_types: variadic<:!lit.anytrait<!UnknownDestructibility> element_type> pos_vararg>()
fn take_many_things_of_specified_trait[element_type: type_of(UnknownDestructibility),
                                       *element_types: element_type]():
    pass


# CHECK-LABEL: lit.fn @"call_many_things_of_specified_trait
fn call_many_things_of_specified_trait(a: TraitStruct):
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:!lit.anytrait<!UnknownDestructibility> !AnyType, :variadic<!AnyType> [!TraitStruct]
    take_many_things_of_specified_trait[AnyType, TraitStruct]()

    # Int is movable.
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:!lit.anytrait<!UnknownDestructibility> !Movable, :variadic<!Movable> [!Int]
    take_many_things_of_specified_trait[Movable, Int]()

    # TraitStruct conforms to SimpleTrait.
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:!lit.anytrait<!UnknownDestructibility> !SimpleTrait, :variadic<!SimpleTrait> [!TraitStruct, !TraitStruct]
    take_many_things_of_specified_trait[SimpleTrait, TraitStruct, TraitStruct]()


comptime _AnyTypeMetaType = type_of(AnyType)

# CHECK-LABEL: lit.struct.decl @TestAnyTrait
struct TestAnyTrait[element_trait: _AnyTypeMetaType]:
    # CHECK: lit.fn @"take_any_type
    # CHECK-SAME: <b_type: !AnyType>[{{.*}}](%self:
    # CHECK-SAME: %b_value: !lit.ref<:!AnyType b_type, imm {{.*}} read_mem)
    fn take_any_type[b_type: AnyType](self, b_value: b_type):
        pass

    # CHECK: lit.fn @"test
    # CHECK-SAME: <a_type: !kgen.param<:!lit.anytrait<!AnyType> element_trait>>
    # CHECK-SAME: (%self: {{.*}}%a_value: !lit.ref<:!kgen.param<:!lit.anytrait<!AnyType> element_trait> a_type, imm {{.*}}> read_mem
    fn test[a_type: Self.element_trait](self, a_value: a_type):
        self.take_any_type(a_value)


@register_passable("trivial")
struct ParamType[x: Int]:
    pass

# CHECK: lit.trait.decl @RGTrait{{.*}} register_passable
@register_passable
trait RGTrait:
    # CHECK-NEXT: lit.fn @"doSomething{{.*}}"[imm *"{{.*}}"](%self: !lit.ref<:!RGTrait *"{{.*}}", imm *"{{.*}}"> read_mem) -> !kgen.none
    fn doSomething(self):
        ...
    # CHECK: lit.fn @"__del__({{.*}})"[mut *"{{.*}}"](%self: !lit.ref<:!RGTrait *"{{.*}}", mut *"{{.*}}"> deinit_mem, |) -> !kgen.none

# CHECK-LABEL: lit.trait.decl @RGTrivialTrait{{.*}} register_passable_trivial
@register_passable("trivial")
trait RGTrivialTrait:
    # CHECK-NEXT: lit.fn @"doSomething{{.*}}"(%self: !kgen.param<:!RGTrivialTrait {{.*}}>) -> !kgen.none
    fn doSomething(self):
        ...


# https://github.com/modular/mojo/issues/3540: Using the output slot breaks trait conformance
# CHECK-LABEL: lit.struct.decl @TestNamedResultConformance
@register_passable("trivial")
struct TestNamedResultConformance(Trait1):

    # CHECK: lit.fn @"f
    # CHECK-SAME: (%self: !TestNamedResultConformance) -> !TestNamedResultConformance
    fn f(self, out output: Self):
        pass

fn test_pack_of_traits1[elt_trait: _AnyTypeMetaType, *elt_types: elt_trait]
                       (var *args: *elt_types):
     pass

fn test_pack_of_traits2[elt_trait: _AnyTypeMetaType, *elt_types: elt_trait](
    var storage: VariadicPack[_, _, elt_trait, *elt_types]):
     pass


comptime _MovableMetaType = type_of(Movable)

fn take_anytype_ref[type: UnknownDestructibility](ref value: type): pass

# CHECK-LABEL: lit.fn @"pass_movable_mt_ref
fn pass_movable_mt_ref[elt_trait: _MovableMetaType, PassT: elt_trait](mut a: PassT):
    # CHECK-NEXT: lit.call @traits::@"take_anytype_ref
    # CHECK-SAME: <:!UnknownDestructibility !kgen.param<:!kgen.param<:!lit.anytrait<!Movable> elt_trait> PassT>
    # CHECK-SAME: : !lit.generator<("value": !lit.ref<:!kgen.param<:!lit.anytrait<!Movable> elt_trait> PassT, mut *"a`"> ref) -> !kgen.none>
    take_anytype_ref(a)

comptime _CollectionElementMetaType = type_of(ImplicitlyCopyable & Movable)

struct FormVariadicPackWithCastedElementVariadic[
    element_trait: _CollectionElementMetaType, //,
    *element_types: element_trait]:

    fn __init__(out self, var *args: *Self.element_types):
        # This should work.
        self.foo(args^)
    fn foo(self, var storage: VariadicPack[_, _, Self.element_trait, *Self.element_types]):
        pass

# This tests that we can take UnsafePointer (which has an AnyType bound for T)
# and conditional conformance rebind the parametric type with AnyType bound down
# to Movable correctly.
fn take_movable_pointer[T: Movable&AnyType](ptr: UnsafePointer[T]): pass
# CHECK-LABEL: test_parametric_anytype_movable
# CHECK-SAME: %ptr: !lit.struct<#UnsafePointer <:!AnyType !kgen.param<:!kgen.param<:!lit.anytrait<!ImplicitlyCopyable_Movable> element_trait>
fn test_parametric_anytype_movable[element_trait: _CollectionElementMetaType,
                                  *element_types: element_trait]
                                  (ptr: UnsafePointer[element_types[0]]):

        # CHECK: lit.call {{.*}}take_movable_pointer
        # CHECK-SAME: <:!Movable_AnyType !kgen.param<:!kgen.param<:!lit.anytrait<!ImplicitlyCopyable_Movable> element_trait>
        take_movable_pointer(ptr)


# This test ensure that overload resolution properly ignores methods coming
# from parent traits when the child trait also has an equivalent definition.

@register_passable("trivial")
trait A:
    fn foo(self: Self):
      pass

@register_passable("trivial")
trait B(A):
    fn foo(self: Self):
      pass

fn blah[b_t: B](b: b_t):
    b.foo()

# Check that a trait method with a default implementation returning None may
# use 'pass'.
trait TBar:
    fn bar(self) -> None:
        pass
