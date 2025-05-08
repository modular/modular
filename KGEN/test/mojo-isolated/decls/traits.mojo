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

    # CHECK: lit.fn @"f2{{.*}}(%self: !lit.ref<{{.*}}> mut) -> !kgen.none attributes
    # CHECK-NEXT: kgen.unreachable
    fn f2(mut self):
        pass

    # CHECK: lit.fn @"f3{{.*}}(%self: !lit.ref<{{.*}}> read_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, mut *"__result__`2x2"> byref_result) throws -> i1
    # CHECK-NEXT: kgen.unreachable
    def f3(self):
        pass

    # CHECK: lit.fn @"f4{{.*}}(%self: !lit.ref<{{.*}}> mut, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, {{.*}}> byref_result) throws -> i1
    # CHECK-NEXT: kgen.unreachable
    def f4(mut self):
        pass

    fn overloaded(self):
        ...

    fn overloaded(self, x: Index):
        ...

    fn overloaded(self, x: string):
        ...

    # CHECK-LABEL: lit.fn @"parametric{{.*}}<x>
    fn parametric[x: Index](self):
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
    # CHECK: lit.fn @"f1{{.*}}"<x: !CFMTraitParams>[{{.*}}](%self: !lit.ref<{{.*}}@CFMStructParams<:type [[T1:.*]], :type [[T2:.*]]>{{.*}}> read_mem)
    fn f1[x: CFMTraitParams](self):
        pass


# ===----------------------------------------------------------------------=== #
# Call Emission
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"generic_trait_fn{{.*}}<T: !Trait>
# CHECK-SAME: %x: !lit.ref<:!Trait T, imm {{.*}}> read_mem
fn generic_trait_fn[T: Trait](x: T):
    # CHECK: lit.call[!lit.generator<[1]("self": {{.*}} read_mem) -> !kgen.none>:
    # CHECK-SAME: get_vtable_entry(:!Trait T, "f0")]{{.*}}(%x)
    x.f0()

    # CHECK: lit.call[!lit.generator<[1]("self": {{[^)]*}}) -> !kgen.none>:
    # CHECK-SAME: get_vtable_entry({{.*}}, "overloaded")]{{.*}}(%x)
    x.overloaded()
    # CHECK: lit.call[!lit.generator<[1]("self": {{.*}}, "x": index)
    # CHECK-SAME: get_vtable_entry({{.*}}, "overloaded")]{{.*}}(%x, %{{.*}})
    x.overloaded(`1`)
    # CHECK: lit.call[!lit.generator<[1]("self": {{.*}}, "x": !kgen.string)
    # CHECK-SAME: get_vtable_entry({{.*}}, "overloaded")]{{.*}}(%x, %{{.*}})
    x.overloaded(__mlir_attr.`"trait" : !kgen.string`)

    # CHECK: lit.call[!lit.generator<[1]("self": {{[^)]*}} read_mem)
    # CHECK-SAME: bind_params(:!lit.generator<<"x": index>[1](
    # CHECK-SAME: get_vtable_entry(:{{.*}} T, "parametric"), 1)
    x.parametric[`1`]()


# CHECK-LABEL: lit.fn @"existential_arg
# CHECK-SAME: (%x: !lit.ref<!Trait, imm {{.*}}>
fn existential_arg(x: Trait):
    pass


trait SimpleTrait:
    fn method(self, y: Index):
        ...

    fn param_method[x: Index](self):
        ...


struct TraitStruct(SimpleTrait):
    fn method(self, y: Index):
        pass

    fn param_method[x: Index](self):
        pass


struct ParametricTraitStruct[z: Index](SimpleTrait):
    fn method(self, y: Index):
        pass

    fn param_method[x: Index](self):
        pass


fn take_simple_trait[T: SimpleTrait]():
    pass


fn infer_trait[T: SimpleTrait](value: T):
    pass


# CHECK-LABEL: lit.fn @"test_metatype_to_trait_vtable
fn test_metatype_to_trait_vtable():
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait [!TraitStruct{{[0-9]*}}, {
    # CHECK-SAME: "method" : !lit.generator<[1]("self": !lit.ref<!TraitStruct, imm {{.*}}> read_mem, "y": index) -> !kgen.none> = {{.*}}@TraitStruct::@"method
    # CHECK-SAME: "param_method" : !lit.generator<<"x": index>[1]("self": !lit.ref<!TraitStruct, imm {{.*}}> read_mem) -> !kgen.none> = {{.*}}@TraitStruct::@"param_method{{.*}}"<?>
    take_simple_trait[TraitStruct]()
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait [{{.*}}@ParametricTraitStruct<2>, {
    # CHECK-SAME: "method" : !lit.generator<[1]("self": {{.*}}@ParametricTraitStruct<2>{{.*}} read_mem, "y": index) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"method{{.*}}"<2>,
    # CHECK-SAME: "param_method" : !lit.generator<<"x": index>[1]("self": {{.*}}@ParametricTraitStruct<2>{{.*}} read_mem) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"param_method{{.*}}"<2, ?>
    take_simple_trait[ParametricTraitStruct[__mlir_attr.`2 : index`]]()


# CHECK-LABEL: lit.fn @"test_infer_trait
fn test_infer_trait(
    a: TraitStruct, b: ParametricTraitStruct[__mlir_attr.`2 : index`]
):
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait [!TraitStruct,
    infer_trait(a)
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait [{{.*}}@ParametricTraitStruct<2>,
    infer_trait(b)


trait StaticMethodTrait:
    @staticmethod
    fn foobar():
        pass


struct StaticMethodStruct(StaticMethodTrait, Copyable):
    @staticmethod
    fn foobar():
        pass

    fn __copyinit__(out self, existing: Self):
        pass


# CHECK-LABEL: lit.fn @"trait_static_method{{.*}}<T: !StaticMethodTrait
fn trait_static_method[T: StaticMethodTrait]():
    # CHECK: call[!lit.generator<() -> !kgen.none>: get_vtable_entry(:!StaticMethodTrait T, "foobar")]()
    T.foobar()


# CHECK-LABEL: lit.fn @"copy_me
# CHECK-SAME: <T: !Copyable
# CHECK-SAME: %value: !lit.ref<:!Copyable T, imm {{.*}}> read_mem, ?,
# CHECK-SAME: %__result__: !lit.ref<:!Copyable T, mut {{.*}}> byref_result
fn copy_me[T: Copyable](value: T) -> T:
    # CHECK-NEXT: call[!lit.generator<[2]("existing": {{.*}}T, {{.*}}> read_mem, |, ?, "self": {{.*}}T, {{.*}}> byref_result) -> !kgen.none>:
    # CHECK-SAME: get_vtable_entry({{.*}} T, "__copyinit__")]{{.*}}(%value, %__result__)
    return value


# CHECK-LABEL: lit.fn @"move_me
# CHECK-SAME: <T: !Movable
# CHECK-SAME: :!Movable T, {{.*}}> owned_in_mem
# CHECK-SAME: :!Movable T, {{.*}}> byref_result
fn move_me[T: Movable](owned value: T) -> T:
    # CHECK-NEXT: lit.ownership.use %value
    # CHECK-NEXT: call[{{.*}}get_vtable_entry({{.*}} T, "__moveinit__")]{{.*}}(%value, %__result__)
    return value^


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    @implicit
    fn __init__(out self, x: Index):
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
    # CHECK-SAME: (%x: index) -> !RegTraitType
    @implicit
    fn __init__(out self, x: Index):
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
    # CHECK: lit.call[{{.*}}: get_vtable_entry(:!TraitForReg T, "may_throw")][{{.*}}](%__error__, %anonymous
    _ = T.may_throw()
    # CHECK: lit.call[{{.*}}: get_vtable_entry(:!TraitForReg T, "throwing_method")][{{.*}}](%{{.*}}, %__error__, %anonymous
    x.throwing_method()


trait CrazyTrait:
    pass

    fn foo[b: Index](self, c: Index) -> Self:
        ...


trait ChangedResultTypeTrait:
    @staticmethod
    fn result_type() -> Self:
        ...


# COM: The calling convention rewrite results in a decl with two "overloads" that
# COM: differ only in result type. Ensure that the thunk gets selected.
@register_passable
struct ChangedResultTypeStruct(ChangedResultTypeTrait):
    @staticmethod
    fn result_type() -> Self:
        pass


# CHECK-LABEL: lit.fn @"convert_result_type
fn convert_result_type():
    @parameter
    fn convert_result_type[T: ChangedResultTypeTrait]():
        pass

    # CHECK: call{{.*}}fn() -> traits::ChangedResultTypeStruct
    convert_result_type[ChangedResultTypeStruct]()


trait SimpleTraitMethod:
    fn foo(self):
        ...


@register_passable
struct VariadicTrait[*I: Index](SimpleTraitMethod):
    fn foo(self):
        pass


# CHECK-LABEL: lit.fn @"test_bind_variadic
fn test_bind_variadic():
    @parameter
    fn bind_trait[T: SimpleTraitMethod]():
        pass

    # CHECK: call
    # CHECK: "foo" : !lit.generator<[1]("self": {{.*}}<:variadic<index> []>{{.*}} read_mem) -> !kgen.none> = {{.*}}@"foo{{.*}}"<:variadic<index> []>
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
    fn take(owned self, owned x: RegTraitType):
        ...


# CHECK-LABEL: lit.struct.decl @NoDtor
@register_passable
struct NoDtor(OwnedArguments, DefaultConstructible):
    fn take(owned self, owned x: RegTraitType):
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
    # CHECK-NEXT: call {{.*}}default_construct{{.*}}<:!DefaultConstructible [!NoDtor,{{.*}}(%c)
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
# CHECK-SAME: (!AnyType_UnknownDestructibility_SimpleTraitA_SimpleTraitB)
@register_passable
struct TwoThunks(SimpleTraitA, SimpleTraitB):
    # CHECK: lit.fn @"method({{.*}}TwoThunks)"
    fn method(self):
        pass


# https://linear.app/modularml/issue/MOCO-335/[bug]-register-passable-generates-phantom-trait-bound-overload
# CHECK-LABEL: lit.fn @"regpassable_reference
fn regpassable_reference():
    # CHECK-NEXT: @TwoThunks::@"method
    alias f = TwoThunks.method


trait RequiredType:
    alias T: AnyType

    @staticmethod
    fn use_it(arg: T) -> T:
        ...


struct RegPassableRequiredType(RequiredType):
    alias T = Int

    @staticmethod
    fn use_it(arg: Int) -> Int:
        pass


# CHECK-LABEL: lit.fn @"bind_regpassable_required_type
fn bind_regpassable_required_type():
    # CHECK-NEXT: fn(::Int) -> ::Int
    # CHECK-SAME: @RegPassableRequiredType::@"use_it
    alias T: RequiredType = RegPassableRequiredType


# ===----------------------------------------------------------------------=== #
# Special Functions
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @RegTrivialSpecial
@register_passable("trivial")
struct RegTrivialSpecial(AnyType, Copyable, Movable):
    pass
    # CHECK: lit.fn @"__del__
    # CHECK: lit.fn @"__copyinit__
    # CHECK: lit.fn @"__moveinit__


# CHECK-LABEL: lit.struct.decl @RegSpecial
@register_passable
struct RegSpecial(AnyType, Copyable, Movable):
    fn __copyinit__(out self, existing: Self):
        pass

    # CHECK: lit.fn @"__del__
    # CHECK: lit.fn @"__moveinit__


# CHECK-LABEL: lit.struct.decl @MemoryOnlySpecial
struct MemoryOnlySpecial(AnyType, Copyable, Movable):
    fn __copyinit__(out self, existing: Self):
        pass

    fn __moveinit__(out self, owned existing: Self):
        pass

    # CHECK: lit.fn @"__del__
    # CHECK-SAME: [{{.*}} owned_in_mem, |) -> !kgen.none
    # CHECK: return %none


fn copy[T: Copyable](x: T):
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

    # CHECK-NOT: foo


# CHECK-LABEL: lit.trait.decl @GreatGrandFather
# CHECK-SAME: (!AnyType_UnknownDestructibility_GreatGrandFather)
trait GreatGrandFather:
    # CHECK: lit.fn @"foo
    fn foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @GrandFather
# CHECK-SAME: GreatGrandFather)
trait GrandFather(GreatGrandFather):
    # CHECK: lit.fn @"bar
    fn bar(self):
        ...

    # CHECK: lit.fn @"foo


# CHECK-LABEL: lit.trait.decl @Father
# CHECK-SAME: GrandFather_GreatGrandFather)
trait Father(GrandFather):
    # CHECK: lit.fn @"baz
    fn baz(self):
        ...

    # CHECK: lit.fn @"bar
    # CHECK: lit.fn @"foo


# CHECK-LABEL: lit.struct.decl @TraitInheritance
# CHECK-SAME: Father_GrandFather_GreatGrandFather)
struct TraitInheritance(Father):
    fn foo(self):
        pass

    fn bar(self):
        pass

    fn baz(self):
        pass


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

    # CHECK: call
    # CHECK-SAME: "foo"
    take_great_grand_father[TraitInheritance]()
    # CHECK: call
    # CHECK-SAME: "bar"
    # CHECK-SAME: "foo"
    take_grand_father[TraitInheritance]()
    # CHECK: call
    # CHECK-SAME: "baz"
    # CHECK-SAME: "bar"
    # CHECK-SAME: "foo"
    take_father[TraitInheritance]()


fn infer_grand_father[T: GrandFather](x: T):
    pass


# CHECK-LABEL: lit.fn @"pass_up_trait
# CHECK-SAME: <T: !Father>
fn pass_up_trait[T: Father](x: T):
    # CHECK-NEXT: call {{.*}}infer_grand_father{{.*}}<:!GrandFather
    # CHECK-SAME: [!kgen.param<:!Father T>, {
    # CHECK-SAME: "bar" : !lit.generator<[1]("self": !lit.ref<:!Father T, imm {{.*}}> read_mem) -> !kgen.none> = get_vtable_entry({{.*}} T, "bar"),
    # CHECK-SAME: "foo" : !lit.generator<[1]("self": !lit.ref<:!Father T, imm {{.*}}> read_mem) -> !kgen.none> = get_vtable_entry({{.*}} T, "foo")
    # CHECK-SAME: }]>(%x)
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
    var x: MovableType[T]


@register_passable("trivial")
struct Item(InCollection):
    pass


fn take_movable(x: MovableType[Item]):
    pass


# CHECK-LABEL: lit.fn @"converted_metatype_struct_element
fn converted_metatype_struct_element(x: Collection[Item]):
    # CHECK: call {{.*}}take_movable{{.*}}"__moveinit__" : {{.*}} = rebind({{.*}}__moveinit__({{.*}})
    take_movable(x.x)


# CHECK-LABEL: lit.struct.decl @TraitMember
# CHECK-NEXT: destructor
struct TraitMember[T: Movable]:
    # CHECK: lit.fn @"__del__
    var value: T


# COM: Misleading error about thunk functions when: (issue mojo-#1402)
#      the test has
#      - a struct conforms to a trait, e.g. Movable
#      - the struct has a field of another type with parameter as itself, e.g MyPointer[Self]
#      - the field struct type's parameter should conform to Movable


# CHECK-LABEL: lit.struct.decl @MyPointer
@value
struct MyPointer[T: AnyType]:
    pass
    # CHECK: lit.fn @"__del__
    # CHECK: lit.fn @"__init__


# CHECK-LABEL: lit.struct.decl @HasMyPointerSelf
struct HasMyPointerSelf(AnyType):
    # CHECK: lit.struct.field x : !lit.struct<#MyPointer <:!AnyType
    var x: MyPointer[Self]
    # CHECK: lit.fn @"__del__

    fn __moveinit__(out self, owned existing: Self, /):
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
    fn add[x: Index](mut self):
        ...


struct KeysContainer[end: Index](KeysBuilder):
    fn add[x: Index](mut self):
        pass


# CHECK-LABEL: lit.fn @"param_trait
fn param_trait[T: SimpleTrait, value: T]():
    # CHECK-NEXT: apply({{.*}} get_vtable_entry(:!SimpleTrait T, "method"){{.*}} store_to_mem(value), 1)
    alias param = value.method(`1`)
    # CHECK-NEXT: [[VAR:%.*]] = lit.var.decl
    # CHECK-NEXT: [[VALUE:%.*]] = kgen.param.materialize
    # CHECK-NEXT: store [[VALUE]], [[VAR]]
    # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[VAR]]
    # CHECK: call[{{.*}}get_vtable_entry(:!SimpleTrait T, "method"){{.*}}([[IMM]], %index2)
    value.method(`2`)


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
    alias T: Makeable = MakeNamedResult


# ===----------------------------------------------------------------------=== #
# Implicit Conformance
# ===----------------------------------------------------------------------=== #


trait ImplicitConformance:
    fn implicit(self):
        ...


trait ImplicitParent:
    fn parent_method(self):
        ...


trait ImplicitChild(ImplicitParent):
    fn child_method(self):
        ...


# COM: The struct decl is modified to conform to the trait.


# CHECK-LABEL: lit.struct.decl @NoExplicitTraits
# CHECK-SAME: (!AnyType_UnknownDestructibility_ImplicitChild_ImplicitConformance_ImplicitParent)
struct NoExplicitTraits:
    fn implicit(self):
        pass

    fn child_method(self):
        pass

    fn parent_method(self):
        pass


# CHECK-LABEL: lit.struct.decl @ChildFirst
# CHECK-SAME: (!AnyType_UnknownDestructibility_ImplicitChild_ImplicitParent)
struct ChildFirst:
    fn child_method(self):
        pass

    fn parent_method(self):
        pass


# CHECK-LABEL: lit.struct.decl @RegisterPassable
# CHECK-SAME: (!AnyType_Copyable_UnknownDestructibility_ImplicitConformance)
@register_passable
struct RegisterPassable:
    # CHECK: lit.fn @"__copyinit__{{.*}}"
    fn __copyinit__(out self, existing: Self):
        pass

    # CHECK: lit.fn @"implicit{{.*}}"
    fn implicit(self):
        pass


# CHECK-LABEL: lit.fn @"test_implicit_conformance
fn test_implicit_conformance():
    # CHECK-NEXT: !ImplicitConformance = <[!NoExplicitTraits, {"implicit" {{.*}}@NoExplicitTraits::@"implicit
    alias bound0: ImplicitConformance = NoExplicitTraits
    # CHECK-NEXT: !ImplicitConformance = <[!NoExplicitTraits, {"implicit"
    alias bound1: ImplicitConformance = NoExplicitTraits

    # CHECK-NEXT: !ImplicitParent = <[!NoExplicitTraits, {"parent_method"
    alias bound2: ImplicitParent = NoExplicitTraits
    # CHECK-NEXT: !ImplicitChild = <[!NoExplicitTraits, {"child_method" {{.*}} "parent_method"
    alias bound3: ImplicitChild = NoExplicitTraits

    # CHECK-NEXT: !ImplicitChild = <[!ChildFirst, {"child_method" {{.*}} "parent_method"
    alias bound4: ImplicitChild = ChildFirst
    # CHECK-NEXT: !ImplicitParent = <[!ChildFirst, {"parent_method"
    alias bound5: ImplicitParent = ChildFirst

    # CHECK-NEXT: !Copyable = <[!RegisterPassable, {"__copyinit__" {{.*}}@RegisterPassable::@"__copyinit__{{.*}}"
    alias bound6: Copyable = RegisterPassable
    # CHECK-NEXT: !ImplicitConformance = <[!RegisterPassable, {"implicit" {{.*}}@RegisterPassable::@"implicit{{.*}}"
    alias bound7: ImplicitConformance = RegisterPassable


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
fn test_infer_sub_trait[T: OtherEmptyTrait](owned foo: Foo[T], bar: Bar[T]):
    # CHECK: call {{.*}}@Foo::@"infer_sub_trait{{.*}}<:!EmptyTrait [!kgen.param<:!OtherEmptyTrait T>, {{.*}}], :!OtherEmptyTrait T>(%foo, %bar)
    var copy = foo.infer_sub_trait(bar)


# ===----------------------------------------------------------------------=== #
# AnyTrait subtyping
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"test_anytrait_subtyping
# CHECK-SAME: <ty: !lit.anytrait<!AnyType>>
fn test_anytrait_subtyping[ty: __type_of(AnyType)]():
    # Call !lit.anytrait subtyping.
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:!lit.anytrait<!AnyType> !AnyType>()
    test_anytrait_subtyping[AnyType]()
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:!lit.anytrait<!AnyType> !SimpleTrait>()
    test_anytrait_subtyping[SimpleTrait]()


# CHECK-LABEL: lit.fn @"take_many_things_of_specified_trait
# CHECK-SAME: <element_type: !lit.anytrait<!AnyType>,
# CHECK-SAME: element_types: variadic<:!lit.anytrait<!AnyType> element_type> pos_vararg>()
fn take_many_things_of_specified_trait[element_type: __type_of(AnyType),
                                       *element_types: element_type]():
    pass


# CHECK-LABEL: lit.fn @"call_many_things_of_specified_trait
fn call_many_things_of_specified_trait(a: TraitStruct):
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> {{.}}[!TraitStruct
    take_many_things_of_specified_trait[AnyType, TraitStruct]()

    # Int is movable.
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:!lit.anytrait<!AnyType> !Movable, :variadic<!Movable> {{.}}[!Int
    take_many_things_of_specified_trait[Movable, Int]()

    # TraitStruct conforms to SimpleTrait.
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:!lit.anytrait<!AnyType> !SimpleTrait, :variadic<!SimpleTrait> {{.}}[!TraitStruct
    take_many_things_of_specified_trait[SimpleTrait, TraitStruct, TraitStruct]()


alias _AnyTypeMetaType = __type_of(AnyType)

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
    fn test[a_type: element_trait](self, a_value: a_type):
        self.take_any_type(a_value)


@register_passable("trivial")
struct ParamType[x: Index]:
    pass

# CHECK: lit.trait.decl @RGTrait{{.*}} register_passable
@register_passable
trait RGTrait:
    # CHECK-NEXT: lit.fn @"doSomething{{.*}}"[imm *"{{.*}}"](%self: !lit.ref<:!RGTrait *"{{.*}}", imm *"{{.*}}"> read_mem) -> !kgen.none
    fn doSomething(self):
        ...
    # CHECK: lit.fn @"__del__({{.*}})"[mut *"{{.*}}"](%self: !lit.ref<:!RGTrait *"{{.*}}", mut *"{{.*}}"> owned_in_mem, |) -> !kgen.none

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
                       (owned *args: *elt_types):
     pass

fn test_pack_of_traits2[elt_trait: _AnyTypeMetaType, *elt_types: elt_trait](
    owned storage: VariadicPack[_, _, elt_trait, *elt_types]):
     pass


alias _MovableMetaType = __type_of(Movable)

fn take_anytype_ref[type: AnyType](ref value: type): pass

# CHECK-LABEL: lit.fn @"pass_movable_mt_ref
fn pass_movable_mt_ref[elt_trait: _MovableMetaType, PassT: elt_trait](mut a: PassT):
    # CHECK-NEXT: lit.call @traits::@"take_anytype_ref
    # CHECK-SAME: <:!AnyType [!kgen.param<:!kgen.param<:!lit.anytrait<!Movable> elt_trait> PassT>, {
    # CHECK-SAME: "__del__" : !lit.generator<[1]("self": !lit.ref<:!kgen.param<:!lit.anytrait<!Movable> elt_trait> PassT, mut *[0,0]> owned_in_mem, |) -> !kgen.none>
    # CHECK-SAME: = get_vtable_entry(:!Movable upcast(:!kgen.param<:!lit.anytrait<!Movable> elt_trait> PassT), "__del__")}], :i1 1, :origin<1> *"a`">(%a)
    # CHECK-SAME: : !lit.generator<("value": !lit.ref<:!kgen.param<:!lit.anytrait<!Movable> elt_trait> PassT, mut *"a`"> ref) -> !kgen.none>
    take_anytype_ref(a)

alias _CollectionElementMetaType = __type_of(Copyable & Movable)

struct FormVariadicPackWithCastedElementVariadic[
    element_trait: _CollectionElementMetaType, //,
    *element_types: element_trait]:

    fn __init__(out self, owned *args: *element_types):
        # This should work.
        self.foo(args^)
    fn foo(self, owned storage: VariadicPack[_, _, element_trait, *element_types]):
        pass

# This tests that we can take UnsafePointer (which has an AnyType bound for T)
# and conditional conformance rebind the parametric type with AnyType bound down
# to Movable correctly.
fn take_movable_pointer[T: Movable](ptr: UnsafePointer[T]): pass
# CHECK-LABEL: test_parametric_anytype_movable
# CHECK-SAME: %ptr: !lit.struct<#UnsafePointer <:!AnyType [!kgen.param<:!kgen.param<:!lit.anytrait<!Copyable_Movable> element_trait>
fn test_parametric_anytype_movable[element_trait: _CollectionElementMetaType,
                                  *element_types: element_trait]
                                  (ptr: UnsafePointer[element_types[0]]):

        # CHECK: lit.call {{.*}}take_movable_pointer
        # CHECK-SAME: <:!Movable [!kgen.param<:!kgen.param<:!lit.anytrait<!Copyable_Movable> element_trait>
        take_movable_pointer(ptr)
