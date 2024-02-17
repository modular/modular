# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-vtables | FileCheck %s


# CHECK-LABEL: lit.trait.decl @Trait<?, MT: type, T: !kgen.paramref<MT>>
trait Trait:
    # CHECK: lit.func @"f0($1)"[{{.*}}](%self: !lit.ref<:!kgen.paramref<MT> T, imm {{.*}}> borrow_in_mem) -> !kgen.none
    # CHECK-NEXT: lit.trait_func
    fn f0(self: Self):
        ...

    # CHECK: lit.func @"f1($1&)"{{.*}}(%self: !lit.ref<:!kgen.paramref<MT> T, mut {{.*}}> byref) -> !kgen.none
    # CHECK-NEXT: lit.trait_func
    fn f1(inout self: Self):
        ...

    # CHECK: lit.func @"f2($1&)"{{.*}}(%self: !lit.ref<:!kgen.paramref<MT> T, mut {{.*}}> byref) -> !kgen.none attributes
    # CHECK-NEXT: lit.trait_func
    fn f2(inout self: Self):
        pass

    # CHECK: lit.func @"f3(,$1)"[{{.*}}](%__result__: !lit.ref<!object, mut {{.*}}> byref_result, |, %self: !lit.ref<:!kgen.paramref<MT> T, mut {{.*}}> owned_in_mem)
    # CHECK-NEXT: lit.trait_func
    def f3(self: Self):
        pass

    # CHECK: lit.func @"f4(,$1&)"[{{.*}}](%__result__: !lit.ref<!object, mut {{.*}}> byref_result, |, %self: !lit.ref<:!kgen.paramref<MT> T, mut {{.*}}> byref)
    # CHECK-NEXT: lit.trait_func
    def f4(inout self: Self):
        pass

    fn overloaded(self):
        ...

    fn overloaded(self, x: int):
        ...

    fn overloaded(self, x: string):
        ...

    # CHECK-LABEL: lit.func @"parametric{{.*}}<x>
    fn parametric[x: int](self):
        ...


# CHECK-LABEL: lit.trait.decl @EmptyTrait<?, MT: type, T: !kgen.paramref<MT>>
trait EmptyTrait:
    pass


# CHECK-LABEL: lit.trait.decl @Trait1<?, MT: type, T: !kgen.paramref<MT>>
trait Trait1:
    # CHECK: lit.func @"f{{.*}}(%__result__: !lit.ref<:!kgen.paramref<MT> T, mut {{.*}}> byref_result, |, %self: !lit.ref<:!kgen.paramref<MT> T, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn f(self: Self) -> Self:
        ...


# CHECK-LABEL: lit.trait.decl @Trait2<?, MT: type, T: !kgen.paramref<MT>>
trait Trait2:
    # CHECK: lit.func @"f{{.*}}(%__result__: !lit.ref<:!kgen.paramref<MT> T, mut {{.*}}> byref_result, |, %self: !lit.ref<:!kgen.paramref<MT> T, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn f(self: Self) -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @StructWithTraits(!Trait1_, {{.*}}, !Trait2_)
struct StructWithTraits(Trait1, Trait2):
    # CHECK: lit.func @"f{{.*}}(%{{.*}}: !lit.ref<!StructWithTraits, mut {{.*}}> byref_result, |, %self: !lit.ref<!StructWithTraits, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn f(self: Self) -> Self:
        ...


# CHECK-LABEL: lit.trait.decl @CFMTrait<?, MT: type, T: !kgen.paramref<MT>>
trait CFMTrait:
    # CHECK: lit.func @"f1{{.*}}(%self: !lit.ref<:!kgen.paramref<MT> T, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn f1(self: Self):
        pass

    # CHECK: lit.func @"f2()"() -> !kgen.none
    @staticmethod
    fn f2():
        pass


# CHECK-LABEL: lit.struct.decl @CFMStruct(!CFMTrait
struct CFMStruct(CFMTrait):
    # CHECK: lit.func @"f1({{.*}})"[{{.*}}](%self: !lit.ref<!CFMStruct, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn f1(self: Self):
        pass

    # CHECK: lit.func @"f2()"() -> !kgen.none
    @staticmethod
    fn f2():
        pass


# Test for struct with parameters and function with parameters.
# CHECK-LABEL: lit.trait.decl @CFMTraitParams<?, MT: type, T: !kgen.paramref<MT>>
trait CFMTraitParams:
    # CHECK: lit.func @"f1{{.*}}"[{{.*}}]<x: !CFMTraitParams>(%self: !lit.ref<:!kgen.paramref<MT> T, imm {{.*}}> borrow_in_mem)
    fn f1[x: CFMTraitParams](self):
        pass


# CHECK-LABEL: lit.struct.decl @CFMStructParams
struct CFMStructParams[t1: AnyRegType, t2: AnyRegType](CFMTraitParams):
    # CHECK: lit.func @"f1{{.*}}"[{{.*}}]<x: !CFMTraitParams>(%self: !lit.ref<{{.*}}@CFMStructParams<:type [[T1:.*]], :type [[T2:.*]]>{{.*}}> borrow_in_mem)
    fn f1[x: CFMTraitParams](self):
        pass


# CHECK-LABEL: lit.func @"generic_trait_fn{{.*}}<T: !Trait>
# CHECK-SAME: %x: !lit.ref<:!Trait T, imm {{.*}}> borrow_in_mem
fn generic_trait_fn[T: Trait](x: T):
    # CHECK: call_param[!lit.signature<[1]("self": {{.*}} borrow_in_mem) -> !kgen.none>:
    # CHECK-SAME: get_type_method(:!Trait T, "f0")]{{.*}}(%x)
    x.f0()

    # CHECK: call_param[!lit.signature<[1]("self": {{[^)]*}}) -> !kgen.none>:
    # CHECK-SAME: get_type_method({{.*}}, "overloaded")]{{.*}}(%x)
    x.overloaded()
    # CHECK: call_param[!lit.signature<[1]("self": {{.*}}, "x": index borrow)
    # CHECK-SAME: get_type_method({{.*}}, "overloaded")]{{.*}}(%x, %{{.*}})
    x.overloaded(`1`)
    # CHECK: call_param[!lit.signature<[1]("self": {{.*}}, "x": !kgen.string borrow)
    # CHECK-SAME: get_type_method({{.*}}, "overloaded")]{{.*}}(%x, %{{.*}})
    x.overloaded(__mlir_attr.`"trait" : !kgen.string`)

    # CHECK: call_param[!lit.signature<[1]("self": {{[^)]*}} borrow_in_mem)
    # CHECK-SAME: bind_signature(:!lit.signature<[1]<"x": index>(
    # CHECK-SAME: get_type_method(:{{.*}} T, "parametric"), 1)
    x.parametric[`1`]()


# CHECK-LABEL: lit.func @"existential_arg
# CHECK-SAME: (%x: !lit.ref<!Trait, imm {{.*}}>
fn existential_arg(x: Trait):
    pass


trait SimpleTrait:
    fn method(self, y: int):
        ...

    fn param_method[x: int](self):
        ...


struct TraitStruct(SimpleTrait):
    fn method(self, y: int):
        pass

    fn param_method[x: int](self):
        pass


struct ParametricTraitStruct[z: int](SimpleTrait):
    fn method(self, y: int):
        pass

    fn param_method[x: int](self):
        pass


fn take_simple_trait[T: SimpleTrait]():
    pass


fn infer_trait[T: SimpleTrait](value: T):
    pass


# CHECK-LABEL: lit.func @"test_metatype_to_trait_vtable
fn test_metatype_to_trait_vtable():
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait [!TraitStruct{{[0-9]*}}, {
    # CHECK-SAME: "method" : !lit.signature<[1]("self": !lit.ref<!TraitStruct, imm {{.*}}> borrow_in_mem, "y": index borrow) -> !kgen.none> = {{.*}}@TraitStruct::@"method
    # CHECK-SAME: "param_method" : !lit.signature<[1]<"x": index>("self": !lit.ref<!TraitStruct, imm {{.*}}> borrow_in_mem) -> !kgen.none> = {{.*}}@TraitStruct::@"param_method{{.*}}"<?>
    take_simple_trait[TraitStruct]()
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait [{{.*}}@ParametricTraitStruct<2> : metatype<{{.*}}>, {
    # CHECK-SAME: "method" : !lit.signature<[1]("self": {{.*}}@ParametricTraitStruct<2>{{.*}} borrow_in_mem, "y": index borrow) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"method{{.*}}"<2>,
    # CHECK-SAME: "param_method" : !lit.signature<[1]<"x": index>("self": {{.*}}@ParametricTraitStruct<2>{{.*}} borrow_in_mem) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"param_method{{.*}}"<2, ?>
    take_simple_trait[ParametricTraitStruct[__mlir_attr.`2 : index`]]()


# CHECK-LABEL: lit.func @"test_infer_trait
fn test_infer_trait(
    a: TraitStruct, b: ParametricTraitStruct[__mlir_attr.`2 : index`]
):
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait [!TraitStruct,
    infer_trait(a)
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait [{{.*}}@ParametricTraitStruct<2> : metatype<{{.*}}>,
    infer_trait(b)


trait StaticMethodTrait:
    @staticmethod
    fn foobar():
        pass


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
        pass


struct StaticMethodStruct(StaticMethodTrait, Copyable):
    @staticmethod
    fn foobar():
        pass

    fn __copyinit__(inout self, existing: Self):
        pass


# CHECK-LABEL: lit.func @"trait_static_method{{.*}}<T: !StaticMethodTrait
fn trait_static_method[T: StaticMethodTrait]():
    # CHECK: call_param[!lit.signature<() -> !kgen.none>: get_type_method(:!StaticMethodTrait T, "foobar")]()
    T.foobar()


# CHECK-LABEL: lit.func @"copy_me
# CHECK-SAME: <T: !Copyable
# CHECK-SAME: %__result__: !lit.ref<:!Copyable T, mut {{.*}}> byref_result, |,
# CHECK-SAME: %value: !lit.ref<:!Copyable T, imm {{.*}}> borrow_in_mem)
fn copy_me[T: Copyable](value: T) -> T:
    # CHECK-NEXT: call_param[!lit.signature<[2]("self": {{.*}}T, {{.*}}> init_self, "existing": {{.*}}T, {{.*}}> borrow_in_mem, |) -> !kgen.none>:
    # CHECK-SAME: get_type_method({{.*}} T, "__copyinit__")]{{.*}}(%__result__, %value)
    return value


# CHECK-LABEL: lit.func @"move_me
# CHECK-SAME: <T: !Movable
# CHECK-SAME: !Movable T, {{.*}}> byref_result
# CHECK-SAME: !Movable T, {{.*}}> owned_in_mem
fn move_me[T: Movable](owned value: T) -> T:
    # CHECK-NEXT: %value28transfer29 = lit.transfer_mem_ownership %value
    # CHECK-NEXT: call_param[{{.*}}get_type_method({{.*}} T, "__moveinit__")]{{.*}}(%__result__, %value28transfer29)
    return value ^


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    fn __init__(inout self, x: int):
        ...

    fn __copyinit__(inout self, existing: Self):
        ...

    @staticmethod
    fn may_throw() raises -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @RegTraitType
@register_passable
struct RegTraitType(TraitForReg):
    # CHECK-LABEL: lit.func @"__init__{{.*}}_thunk"
    # CHECK-SAME: %self: !lit.ref<!RegTraitType, mut {{.*}}> init_self, |, %x: index borrow) -> !kgen.none
    fn __init__(x: int) -> Self:
        # CHECK: %0 = lit.call {{.*}}@RegTraitType{{.*}}__init__{{.*}}(%x)
        # CHECK: store %0, %self
        pass

    # CHECK-LABEL: lit.func @"__copyinit__{{.*}}_thunk"
    # CHECK-SAME: %self: !lit.ref<!RegTraitType, mut {{.*}}> init_self, |, %arg[existing]: !lit.ref<!RegTraitType, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn __copyinit__(existing: Self) -> Self:
        # CHECK: %0 = lit.ref.load %arg
        # CHECK: %1 = lit.call {{.*}}@RegTraitType{{.*}}__copyinit__{{.*}}(%0)
        # CHECK: store %1, %self
        pass

    # CHECK-LABEL: lit.func @"may_throw{{.*}}_thunk"
    # CHECK-SAME: %__result__: !lit.ref<!RegTraitType, mut {{.*}}> byref_result
    # CHECK-SAME: throws|ownedresult -> !kgen.variant<!Error, none> always_inline
    @staticmethod
    fn may_throw() raises -> Self:
        # CHECK: %0 = lit.call {{.*}}@RegTraitType::@"may_throw()"
        # CHECK: %1 = lit.handle_variant %0
        # CHECK: store %1, %__result__
        pass


trait CrazyTrait:
    pass

    fn foo[b: int](self, c: int) -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @CrazyRegisterPassable<a>
@value
@register_passable
struct CrazyRegisterPassable[a: int](CrazyTrait):
    pass

    # CHECK-LABEL: lit.func @"foo{{.*}}_thunk"
    # CHECK-SAME: <b>(%__result__: !lit.ref<{{.*}}@CrazyRegisterPassable<a>{{.*}} byref_result, |,
    # CHECK-SAME: %self: !lit.ref<{{.*}}@CrazyRegisterPassable<a>{{.*}} borrow_in_mem
    # CHECK-SAME: %c: index borrow) -> !kgen.none
    fn foo[b: int](self, c: int) -> Self:
        # CHECK: %0 = lit.ref.load %self
        # CHECK: %1 = lit.call {{.*}}@CrazyRegisterPassable::@"foo{{.*}}<a, b>(%0, %c)
        # CHECK: lit.ref.store %1, %__result__
        return self


@register_passable
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


# CHECK-LABEL: lit.func @"convert_result_type
fn convert_result_type():
    @parameter
    fn convert_result_type[T: ChangedResultTypeTrait]():
        pass

    # CHECK: call_param{{.*}}@ChangedResultTypeStruct::@"result_type()_thunk"
    convert_result_type[ChangedResultTypeStruct]()


trait SimpleTraitMethod:
    fn foo(self):
        ...


@register_passable
struct VariadicTrait[*I: int](SimpleTraitMethod):
    fn foo(self):
        pass


# CHECK-LABEL: lit.func @"test_bind_variadic
fn test_bind_variadic():
    @parameter
    fn bind_trait[T: SimpleTraitMethod]():
        pass

    # CHECK: call_param
    # CHECK: "foo" : !lit.signature<[1]("self": {{.*}}<:variadic<index> []>{{.*}} borrow_in_mem) -> !kgen.none> = {{.*}}@"foo{{.*}}_thunk"<:variadic<index> []>
    bind_trait[VariadicTrait[]]()


trait ThunkAmbiguity:
    fn mismatched_arg(self):
        ...

    @staticmethod
    fn mismatched_ret() -> Self:
        ...

    fn __init__(inout self):
        ...


@register_passable
struct ThunkAmbiguityRP(ThunkAmbiguity):
    fn mismatched_arg(self):
        pass

    @staticmethod
    fn mismatched_ret() -> Self:
        return Self {}

    fn __init__() -> Self:
        return Self {}


# COM: Make sure that the generated thunks aren't select over the methods.


# CHECK-LABEL: lit.func @"ambiguous_thunk
fn ambiguous_thunk(x: ThunkAmbiguityRP):
    # CHECK-NOT: `thunk
    x.mismatched_arg()
    _ = ThunkAmbiguityRP.mismatched_ret()
    _ = ThunkAmbiguityRP()
    # CHECK-LABEL: lit.end_func


trait OwnedArguments:
    fn take(owned self, owned x: RegTraitType):
        ...


# CHECK-LABEL: lit.struct.decl @NoDtor
@register_passable
struct NoDtor(OwnedArguments, DefaultConstructible):
    # CHECK-LABEL: lit.func @"take{{.*}}_thunk"
    fn take(owned self, owned x: RegTraitType):
        # CHECK-NEXT: %0 = lit.load.consume %self
        # CHECK-NEXT: lit.call {{.*}}take{{.*}}(%0, %x)
        pass

    fn __init__() -> Self:
        pass

    fn method(self):
        pass


trait DefaultConstructible:
    fn __init__(inout self):
        ...


fn default_construct[T: DefaultConstructible]() -> T:
    return T()


# CHECK-LABEL: lit.func @"generic_fn_return_type
fn generic_fn_return_type():
    # CHECK: lit.var.decl "c" var : !lit.ref<!NoDtor,
    # CHECK-NEXT: call {{.*}}default_construct{{.*}}<:!DefaultConstructible [!NoDtor,{{.*}}(%c)
    let c = default_construct[NoDtor]()
    # CHECK: call {{.*}}@NoDtor::@"method
    c.method()


# CHECK-LABEL: lit.struct.decl @RegTrivialSpecial
@register_passable("trivial")
struct RegTrivialSpecial(AnyType, Copyable, Movable):
    pass
    # CHECK: lit.func @"__del__{{.*}}_thunk"
    # CHECK-SAME: %0[{{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug
    # CHECK: return %none

    # CHECK: lit.func @"__copyinit__{{.*}}_thunk"
    # CHECK-SAME: %0[{{.*}} init_self, %1[{{.*}} borrow_in_mem
    # CHECK-NEXT: [[V:%.*]] = lit.ref.load %1
    # CHECK-NEXT: lit.ref.store [[V]], %0

    # CHECK: lit.func @"__moveinit__{{.*}}_thunk"{{.*}}(%0[{{.*}} init_self, %1[{{.*}} owned_in_mem
    # CHECK: [[V:%.*]] = lit.load.consume
    # CHECK-NEXT: lit.ref.store [[V]], %0


# CHECK-LABEL: lit.struct.decl @RegSpecial
@register_passable
struct RegSpecial(AnyType, Copyable, Movable):
    fn __copyinit__(existing: Self) -> Self:
        return Self {}

    # CHECK: lit.func @"__del__{{.*}}_thunk"
    # CHECK-SAME: {{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug
    # CHECK: return %none

    # CHECK: lit.func @"__moveinit__{{.*}}_thunk"
    # CHECK-SAME: %0[{{.*}} init_self, %1[{{.*}} owned_in_mem
    # CHECK-NEXT: [[V:%.*]] = lit.load.consume %1
    # CHECK-NEXT: lit.ref.store [[V]], %0


# CHECK-LABEL: lit.struct.decl @MemoryOnlySpecial
struct MemoryOnlySpecial(AnyType, Copyable, Movable):
    fn __copyinit__(inout self, existing: Self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    # CHECK: lit.func @"__del__
    # CHECK-SAME: [{{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug
    # CHECK: return %none


fn copy[T: Copyable](x: T):
    pass


fn move[T: Movable](x: T):
    pass


fn destroy[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.func @"test_special_fn_traits
fn test_special_fn_traits(
    inout x: RegTrivialSpecial, inout y: RegSpecial, inout z: MemoryOnlySpecial
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


trait ParentTraitSameSig:
    fn foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @ChildTraitSameSig
trait ChildTraitSameSig(ParentTraitSameSig):
    # CHECK-NEXT: lit.func @"foo
    # CHECK-NEXT: lit.trait_func
    fn foo(self):
        ...

    # CHECK-NOT: foo


# CHECK-LABEL: lit.trait.decl @GreatGrandFather
# CHECK-SAME: (!AnyType)
trait GreatGrandFather:
    # CHECK: lit.func @"foo
    fn foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @GrandFather
# CHECK-SAME: (!GreatGrandFather,
trait GrandFather(GreatGrandFather):
    # CHECK: lit.func @"bar
    fn bar(self):
        ...

    # CHECK: lit.func @"foo


# CHECK-LABEL: lit.trait.decl @Father
# CHECK-SAME: (!GrandFather, !GreatGrandFather[!GrandFather],
trait Father(GrandFather):
    # CHECK: lit.func @"baz
    fn baz(self):
        ...

    # CHECK: lit.func @"bar
    # CHECK: lit.func @"foo


# CHECK-LABEL: lit.struct.decl @TraitInheritance
# CHECK-SAME: (!Father, !GrandFather[!Father], !GreatGrandFather[!GrandFather, !Father],
struct TraitInheritance(Father):
    fn foo(self):
        pass

    fn bar(self):
        pass

    fn baz(self):
        pass


# CHECK-LABEL: lit.func @"test_trait_inheritance
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

    # CHECK: call_param
    # CHECK-SAME: "foo"
    take_great_grand_father[TraitInheritance]()
    # CHECK: call_param
    # CHECK-SAME: "bar"
    # CHECK-SAME: "foo"
    take_grand_father[TraitInheritance]()
    # CHECK: call_param
    # CHECK-SAME: "baz"
    # CHECK-SAME: "bar"
    # CHECK-SAME: "foo"
    take_father[TraitInheritance]()


fn infer_grand_father[T: GrandFather](x: T):
    pass


# CHECK-LABEL: lit.func @"pass_up_trait
# CHECK-SAME: <T: !Father>
fn pass_up_trait[T: Father](x: T):
    # CHECK-NEXT: call {{.*}}infer_grand_father{{.*}}<:!GrandFather
    # CHECK-SAME: [!kgen.paramref<:!Father T>, {
    # CHECK-SAME: "bar" : !lit.signature<[1]("self": !lit.ref<:!Father T, imm {{.*}}> borrow_in_mem) -> !kgen.none> = get_type_method({{.*}} T, "bar"),
    # CHECK-SAME: "foo" : !lit.signature<[1]("self": !lit.ref<:!Father T, imm {{.*}}> borrow_in_mem) -> !kgen.none> = get_type_method({{.*}} T, "foo")
    # CHECK-SAME: }]>(%x)
    infer_grand_father(x)


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


# CHECK-LABEL: lit.func @"converted_metatype_struct_element
fn converted_metatype_struct_element(x: Collection[Item]):
    # CHECK: call {{.*}}take_movable{{.*}}"__moveinit__" : {{.*}} = rebind({{.*}}__moveinit__({{.*}})_thunk
    take_movable(x.x)


# CHECK-LABEL: lit.struct.decl @TraitMember
# CHECK-SAME: destructor =
struct TraitMember[T: Movable]:
    # CHECK: lit.func @"__del__
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
    # CHECK: lit.func @"__del__
    # CHECK: lit.func @"__init__


# CHECK-LABEL: lit.struct.decl @HasMyPointerSelf
struct HasMyPointerSelf(AnyType):
    # CHECK: lit.struct.field x : !lit.declref<#MyPointer <:!AnyType
    var x: MyPointer[Self]
    # CHECK: lit.func @"__del__

    fn __moveinit__(inout self, owned existing: Self, /):
        pass


# Parser crash
# https://github.com/modularml/modular/issues/27897
# CHECK-LABEL: lit.func @"check_trait_conversion_bymem_result_alias_crash
fn retMemory[T: TraitForReg](value: T) -> MemoryOnlySpecial:
    pass


fn check_trait_conversion_bymem_result_alias_crash(
    x: RegTraitType,
) -> MemoryOnlySpecial:
    return retMemory(x)


# Calling functions with implicit lifetimes needs to cooperate.
fn test[a: ABC]():
    _ = ABCOptionalParamInt[ABCDim(a)]()


trait SomeTrait:
    pass


struct ABC(SomeTrait):
    fn __init__(inout self):
        pass


@register_passable("trivial")
struct ABCOptionalParamInt[dim_parametric: ABCDim]:
    fn __init__() -> Self:
        pass


struct ABCDim:
    fn __init__[type: SomeTrait](inout self, value: type):
        pass
