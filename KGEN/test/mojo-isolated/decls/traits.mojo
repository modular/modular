# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -verify-parameters --kgen-print-inline-type-values | FileCheck %s


# CHECK-LABEL: lit.trait.decl @Trait
# CHECK-SAME: <?, [[T:.*]]: !Trait>
trait Trait:
    # CHECK: lit.func @"f0{{.*}}(%self: !lit.ref<:!Trait [[T]], imm {{.*}}> borrow_in_mem) -> !kgen.none
    # CHECK-NEXT: lit.trait_func
    fn f0(self: Self):
        ...

    # CHECK: lit.func @"f1{{.*}}(%self: !lit.ref<{{.*}}> inout) -> !kgen.none
    # CHECK-NEXT: lit.trait_func
    fn f1(inout self: Self):
        ...

    # CHECK: lit.func @"f2{{.*}}(%self: !lit.ref<{{.*}}> inout) -> !kgen.none attributes
    # CHECK-NEXT: lit.trait_func
    fn f2(inout self: Self):
        pass

    # CHECK: lit.func @"f3{{.*}}(%self: !lit.ref<{{.*}}> borrow_in_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<!object, {{.*}}> byref_result) throws -> i1
    # CHECK-NEXT: lit.trait_func
    def f3(self: Self):
        pass

    # CHECK: lit.func @"f4{{.*}}(%self: !lit.ref<{{.*}}> inout, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<!object, {{.*}}> byref_result) throws -> i1
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


# CHECK-LABEL: lit.trait.decl @EmptyTrait
trait EmptyTrait:
    pass


# CHECK-LABEL: lit.trait.decl @Trait1
# CHECK-SAME: <?, [[T:.*]]: !Trait1_>
trait Trait1:
    # CHECK: lit.func @"f{{.*}}(%self: !lit.ref<{{.*}}> borrow_in_mem, ?, %__result__: !lit.ref<:!Trait1_ [[T]], mut {{.*}}> byref_result) -> !kgen.none
    fn f(self: Self) -> Self:
        ...


trait Trait2:
    fn f(self: Self) -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @StructWithTraits(!Trait1_, {{.*}}, !Trait2_)
struct StructWithTraits(Trait1, Trait2):
    # CHECK: lit.func @"f{{.*}}(%self: !lit.ref<!StructWithTraits, imm {{.*}}> borrow_in_mem, ?, %{{.*}}: !lit.ref<!StructWithTraits, mut {{.*}}> byref_result) -> !kgen.none
    fn f(self: Self) -> Self:
        ...


# CHECK-LABEL: lit.trait.decl @CFMTrait
trait CFMTrait:
    # CHECK: lit.func @"f1{{.*}}(%self: !lit.ref<{{.*}}> borrow_in_mem) -> !kgen.none
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
# CHECK-LABEL: lit.trait.decl @CFMTraitParams
trait CFMTraitParams:
    # CHECK: lit.func @"f1{{.*}}"[{{.*}}]<x: !CFMTraitParams>(
    fn f1[x: CFMTraitParams](self):
        pass


# CHECK-LABEL: lit.struct.decl @CFMStructParams
struct CFMStructParams[t1: AnyTrivialRegType, t2: AnyTrivialRegType](
    CFMTraitParams
):
    # CHECK: lit.func @"f1{{.*}}"[{{.*}}]<x: !CFMTraitParams>(%self: !lit.ref<{{.*}}@CFMStructParams<:type [[T1:.*]], :type [[T2:.*]]>{{.*}}> borrow_in_mem)
    fn f1[x: CFMTraitParams](self):
        pass


# ===----------------------------------------------------------------------=== #
# Call Emission
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.func @"generic_trait_fn{{.*}}<T: !Trait>
# CHECK-SAME: %x: !lit.ref<:!Trait T, imm {{.*}}> borrow_in_mem
fn generic_trait_fn[T: Trait](x: T):
    # CHECK-NEXT: [[XI:%.*]] = kgen.rebind %x {{.*}}#lit.invalid.ref.lifetime
    # CHECK: lit.call[!lit.signature<[1]("self": {{.*}} borrow_in_mem) -> !kgen.none>:
    # CHECK-SAME: get_type_method(:!Trait T, "f0")]{{.*}}([[XI]])
    x.f0()

    # CHECK: lit.call[!lit.signature<[1]("self": {{[^)]*}}) -> !kgen.none>:
    # CHECK-SAME: get_type_method({{.*}}, "overloaded")]{{.*}}([[XI]])
    x.overloaded()
    # CHECK: lit.call[!lit.signature<[1]("self": {{.*}}, "x": index)
    # CHECK-SAME: get_type_method({{.*}}, "overloaded")]{{.*}}([[XI]], %{{.*}})
    x.overloaded(`1`)
    # CHECK: lit.call[!lit.signature<[1]("self": {{.*}}, "x": !kgen.string)
    # CHECK-SAME: get_type_method({{.*}}, "overloaded")]{{.*}}([[XI]], %{{.*}})
    x.overloaded(__mlir_attr.`"trait" : !kgen.string`)

    # CHECK: lit.call[!lit.signature<[1]("self": {{[^)]*}} borrow_in_mem)
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
    # CHECK-SAME: "method" : !lit.signature<[1]("self": !lit.ref<!TraitStruct, imm {{.*}}> borrow_in_mem, "y": index) -> !kgen.none> = {{.*}}@TraitStruct::@"method
    # CHECK-SAME: "param_method" : !lit.signature<[1]<"x": index>("self": !lit.ref<!TraitStruct, imm {{.*}}> borrow_in_mem) -> !kgen.none> = {{.*}}@TraitStruct::@"param_method{{.*}}"<?>
    take_simple_trait[TraitStruct]()
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!SimpleTrait [{{.*}}@ParametricTraitStruct<2> : anystruct<{{.*}}>, {
    # CHECK-SAME: "method" : !lit.signature<[1]("self": {{.*}}@ParametricTraitStruct<2>{{.*}} borrow_in_mem, "y": index) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"method{{.*}}"<2>,
    # CHECK-SAME: "param_method" : !lit.signature<[1]<"x": index>("self": {{.*}}@ParametricTraitStruct<2>{{.*}} borrow_in_mem) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"param_method{{.*}}"<2, ?>
    take_simple_trait[ParametricTraitStruct[__mlir_attr.`2 : index`]]()


# CHECK-LABEL: lit.func @"test_infer_trait
fn test_infer_trait(
    a: TraitStruct, b: ParametricTraitStruct[__mlir_attr.`2 : index`]
):
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait [!TraitStruct,
    infer_trait(a)
    # CHECK: call {{.*}}infer_trait{{.*}}<:!SimpleTrait [{{.*}}@ParametricTraitStruct<2> : anystruct<{{.*}}>,
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
    # CHECK: call[!lit.signature<() -> !kgen.none>: get_type_method(:!StaticMethodTrait T, "foobar")]()
    T.foobar()


# CHECK-LABEL: lit.func @"copy_me
# CHECK-SAME: <T: !Copyable
# CHECK-SAME: %value: !lit.ref<:!Copyable T, imm {{.*}}> borrow_in_mem, ?,
# CHECK-SAME: %__result__: !lit.ref<:!Copyable T, mut {{.*}}> byref_result
fn copy_me[T: Copyable](value: T) -> T:
    # CHECK-NEXT: [[VI:%.*]] = kgen.rebind %value {{.*}}#lit.invalid.ref.lifetime
    # CHECK-NEXT: call[!lit.signature<[2]("self": {{.*}}T, {{.*}}> init_self, "existing": {{.*}}T, {{.*}}> borrow_in_mem, |) -> !kgen.none>:
    # CHECK-SAME: get_type_method({{.*}} T, "__copyinit__")]{{.*}}(%__result__, [[VI]])
    return value


# CHECK-LABEL: lit.func @"move_me
# CHECK-SAME: <T: !Movable
# CHECK-SAME: !Movable T, {{.*}}> owned_in_mem
# CHECK-SAME: !Movable T, {{.*}}> byref_result
fn move_me[T: Movable](owned value: T) -> T:
    # CHECK-NEXT: [[VI:%.*]] = kgen.rebind %value {{.*}}#lit.invalid.ref.lifetime
    # CHECK-NEXT: %value28transfer29 = lit.transfer_mem_ownership [[VI]]
    # CHECK-NEXT: call[{{.*}}get_type_method({{.*}} T, "__moveinit__")]{{.*}}(%__result__, %value28transfer29)
    return value^


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    fn __init__(inout self, x: int):
        ...

    fn __copyinit__(inout self, existing: Self):
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
    # CHECK-LABEL: lit.func @"__init__
    # CHECK-SAME: %self: !lit.ref<!RegTraitType, mut {{.*}}> init_self, |, %x: index)
    fn __init__(inout self, x: int):
        pass

    # CHECK-LABEL: lit.func @"__copyinit__{{.*}}_thunk"
    # CHECK-SAME: %self: !lit.ref<!RegTraitType, mut {{.*}}> init_self, |, %existing: !lit.ref<!RegTraitType, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn __copyinit__(existing: Self) -> Self:
        # CHECK: %0 = lit.ref.load %existing
        # CHECK: %1 = lit.call {{.*}}@RegTraitType{{.*}}__copyinit__{{.*}}(%0)
        # CHECK: store %1, %self
        pass

    @staticmethod
    fn may_throw() raises -> Self:
        pass

    # CHECK-LABEL: lit.func @"throwing_method{{.*}}_thunk"
    # CHECK-SAME: (%self: !lit.ref<!RegTraitType, {{.*}} borrow_in_mem, ?, %__error__{{.*}}, %__result__{{.*}})
    fn throwing_method(self) raises:
        # CHECK-NEXT: [[SELF_REG:%.*]] = lit.ref.load %self
        # CHECK-NEXT: lit.call {{.*}}RegTraitType::@"throwing_method{{.*}}([[SELF_REG]], %__error__, %__result__)
        # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
        # CHECK-NEXT: return [[FALSE]]
        pass


# CHECK-LABEL: lit.func @"raising_method
fn raising_method[T: TraitForReg](x: T) raises:
    # CHECK: lit.call[{{.*}}: get_type_method(:!TraitForReg T, "may_throw")][{{.*}}](%__error__, %anonymous
    _ = T.may_throw()
    # CHECK: lit.call[{{.*}}: get_type_method(:!TraitForReg T, "throwing_method")][{{.*}}](%{{.*}}, %__error__, %anonymous
    x.throwing_method()


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
    # CHECK-SAME: <b>(%self: !lit.ref<{{.*}}@CrazyRegisterPassable<a>{{.*}} borrow_in_mem
    # CHECK-SAME: %c: index,
    # CHECK-SAME: %__result__: !lit.ref<{{.*}}@CrazyRegisterPassable<a>{{.*}} byref_result
    fn foo[b: int](self, c: int) -> Self:
        # CHECK: %0 = lit.ref.load %self
        # CHECK: %1 = lit.call {{.*}}@CrazyRegisterPassable::@"foo{{.*}}<a, b>(%0, %c)
        # CHECK: lit.ref.store %1, %__result__
        return self


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

    # CHECK: call{{.*}}@ChangedResultTypeStruct::@"result_type()_thunk"
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

    # CHECK: call
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

    fn __init__(inout self):
        pass


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

    fn __init__(inout self):
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
# CHECK-SAME: (!SimpleTraitA, !AnyType[!SimpleTraitA], !SimpleTraitB)
@register_passable
struct TwoThunks(SimpleTraitA, SimpleTraitB):
    # CHECK: lit.func @"method({{.*}}TwoThunks)"
    # CHECK: lit.func @"method({{.*}}TwoThunks)_thunk"
    fn method(self):
        pass


# ===----------------------------------------------------------------------=== #
# Special Functions
# ===----------------------------------------------------------------------=== #


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
    fn __copyinit__(inout self, existing: Self):
        pass

    # CHECK: lit.func @"__del__{{.*}}_thunk"
    # CHECK-SAME: {{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug

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
    # CHECK-SAME: [{{.*}} owned_in_mem, |) -> !kgen.none
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


# ===----------------------------------------------------------------------=== #
# Inheritance
# ===----------------------------------------------------------------------=== #


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


# CHECK-LABEL: lit.func @"pass_up_trait
# CHECK-SAME: <T: !Father>
fn pass_up_trait[T: Father](x: T):
    # CHECK-NEXT: [[XI:%.*]] = kgen.rebind %x {{.*}}#lit.invalid.ref.lifetime
    # CHECK-NEXT: call {{.*}}infer_grand_father{{.*}}<:!GrandFather
    # CHECK-SAME: [!kgen.paramref<:!Father T>, {
    # CHECK-SAME: "bar" : !lit.signature<[1]("self": !lit.ref<:!Father T, imm {{.*}}> borrow_in_mem) -> !kgen.none> = get_type_method({{.*}} T, "bar"),
    # CHECK-SAME: "foo" : !lit.signature<[1]("self": !lit.ref<:!Father T, imm {{.*}}> borrow_in_mem) -> !kgen.none> = get_type_method({{.*}} T, "foo")
    # CHECK-SAME: }]>([[XI]])
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


# CHECK-LABEL: lit.func @"converted_metatype_struct_element
fn converted_metatype_struct_element(x: Collection[Item]):
    # CHECK: call {{.*}}take_movable{{.*}}"__moveinit__" : {{.*}} = rebind({{.*}}__moveinit__({{.*}})_thunk
    take_movable(x.x)


# CHECK-LABEL: lit.struct.decl @TraitMember
# CHECK-NEXT: destructor
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
    fn __init__(inout self):
        pass


struct ABCDim:
    fn __init__[type: SomeTrait](inout self, value: type):
        pass


trait TraitParameterized:
    fn foo[T: SomeTrait](self):
        ...


struct ConcreteType(TraitParameterized):
    fn foo[T: SomeTrait](self):
        pass


trait KeysBuilder:
    fn add[x: int](inout self):
        ...


struct KeysContainer[end: int](KeysBuilder):
    fn add[x: int](inout self):
        pass


# CHECK-LABEL: lit.func @"param_trait
fn param_trait[T: SimpleTrait, value: T]():
    # CHECK-NEXT: apply({{.*}} get_type_method(:!SimpleTrait T, "method"){{.*}} store_to_mem(value)), 1)
    alias param = value.method(`1`)
    # CHECK-NEXT: [[VAR:%.*]] = lit.var.decl
    # CHECK-NEXT: [[VALUE:%.*]] = kgen.param.materialize
    # CHECK-NEXT: store [[VALUE]], [[VAR]]
    # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[VAR]]
    # CHECK: call[{{.*}}get_type_method(:!SimpleTrait T, "method"){{.*}}([[IMM]], %index2)
    value.method(`2`)


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
# CHECK-SAME: (!AnyType, !ImplicitConformance, !ImplicitParent, !ImplicitChild)
struct NoExplicitTraits:
    fn implicit(self):
        pass

    fn child_method(self):
        pass

    fn parent_method(self):
        pass


# CHECK-LABEL: lit.struct.decl @ChildFirst
# CHECK-SAME: (!AnyType, !ImplicitChild, !ImplicitParent[!ImplicitChild])
struct ChildFirst:
    fn child_method(self):
        pass

    fn parent_method(self):
        pass


# CHECK-LABEL: lit.struct.decl @RegisterPassable
# CHECK-SAME: (!AnyType, !Copyable, !ImplicitConformance)
@register_passable
struct RegisterPassable:
    # CHECK: lit.func @"__copyinit__{{.*}}_thunk"
    fn __copyinit__(existing: Self) -> Self:
        return Self {}

    # CHECK: lit.func @"implicit{{.*}}_thunk"
    fn implicit(self):
        pass


# CHECK-LABEL: lit.func @"test_implicit_conformance
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

    # CHECK-NEXT: !Copyable = <[!RegisterPassable, {"__copyinit__" {{.*}}@RegisterPassable::@"__copyinit__{{.*}}_thunk"
    alias bound6: Copyable = RegisterPassable
    # CHECK-NEXT: !ImplicitConformance = <[!RegisterPassable, {"implicit" {{.*}}@RegisterPassable::@"implicit{{.*}}_thunk"
    alias bound7: ImplicitConformance = RegisterPassable


# COM: Issue https://github.com/modularml/modular/issues/33939
# COM: Ensure parameter inference works between type value attributes.
trait OtherEmptyTrait(EmptyTrait):
    pass


struct Bar[T: EmptyTrait]:
    pass


struct Foo[T: EmptyTrait]:
    fn infer_sub_trait[OT: OtherEmptyTrait](inout self, existing: Bar[OT]):
        pass


# CHECK-LABEL: lit.func @"test_infer_sub_trait
fn test_infer_sub_trait[T: OtherEmptyTrait](owned foo: Foo[T], bar: Bar[T]):
    # CHECK: call {{.*}}@Foo::@"infer_sub_trait{{.*}}<:!EmptyTrait [!kgen.paramref<:!OtherEmptyTrait T>, {{.*}}], :!OtherEmptyTrait T>(%foo, %bar)
    var copy = foo.infer_sub_trait(bar)


# ===----------------------------------------------------------------------=== #
# AnyTrait subtyping
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.func @"test_anytrait_subtyping
# CHECK-SAME: <ty: !lit.anytrait<!AnyType>>
fn test_anytrait_subtyping[ty: __mlir_type[`!lit.anytrait<`, AnyType, `>`]]():
    # Call !lit.anytrait subtyping.
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:!lit.anytrait<!AnyType> !AnyType>()
    test_anytrait_subtyping[AnyType]()
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:!lit.anytrait<!AnyType> !SimpleTrait>()
    test_anytrait_subtyping[SimpleTrait]()


# CHECK-LABEL: lit.func @"take_many_things_of_specified_trait
# CHECK-SAME: <element_type: !lit.anytrait<!AnyType>,
# CHECK-SAME: element_types: variadic<:!lit.anytrait<!AnyType> element_type> var>()
fn take_many_things_of_specified_trait[
    element_type: __mlir_type[`!lit.anytrait<`, AnyType, `>`],
    *element_types: element_type,
]():
    pass


# CHECK-LABEL: lit.func @"call_many_things_of_specified_trait
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


alias _AnyTypeMetaType = __mlir_type[`!lit.anytrait<`, AnyType, `>`]


# CHECK-LABEL: lit.struct.decl @TestAnyTrait
struct TestAnyTrait[element_trait: _AnyTypeMetaType]:
    # CHECK: lit.func @"take_any_type
    # CHECK-SAME: <b_type: !AnyType>(%self:
    # CHECK-SAME: %b_value: !lit.ref<:!AnyType b_type, imm {{.*}} borrow_in_mem)
    fn take_any_type[b_type: AnyType](self, b_value: b_type):
        pass

    # CHECK: lit.func @"test
    # CHECK-SAME: <a_type: !kgen.paramref<:!lit.anytrait<!AnyType> element_trait>>(%self:
    # CHECK-SAME: %a_value: !lit.ref<:!kgen.paramref<:!lit.anytrait<!AnyType> element_trait> a_type, imm {{.*}}> borrow_in_mem
    fn test[a_type: element_trait](self, a_value: a_type):
        self.take_any_type(a_value)


@register_passable("trivial")
struct ParamType[x: int]:
    pass


trait DependentParam:
    fn foo[x: int, y: ParamType[x]](self):
        ...

    fn shadow[a: Int](self):
        ...

    fn bar[x: int, y: int](self, z: ParamType[x]) -> ParamType[y]:
        ...


# CHECK-LABEL: lit.struct.decl @RegPassableParamTrait<a>
@register_passable
struct RegPassableParamTrait[a: int](DependentParam):
    # CHECK: lit.func @"foo{{.*}}_thunk"
    # CHECK-SAME: <x, y: {{.*}}ParamType<x>>(%self: !lit.ref{{.*}}RegPassableParamTrait<a>
    fn foo[x: int, y: ParamType[x]](self):
        # CHECK: call {{.*}}<a, x, :{{.*}}ParamType<x> y>
        pass

    # CHECK: lit.func @"shadow{{.*}}_thunk"
    # CHECK-SAME: <*"a`": !Int>
    fn shadow[b: Int](self):
        # CHECK: call {{.*}}<a, :!Int *"a`">
        pass

    # CHECK: lit.func @"bar{{.*}}_thunk"
    # CHECK-SAME: <x, y>(%self: !lit.ref<{{.*}}RegPassableParamTrait<a>{{.*}}, %z: !lit.declref<#ParamType <x>>) -> !lit.declref<#ParamType <y>>
    fn bar[x: int, y: int](self, z: ParamType[x]) -> ParamType[y]:
        # CHECK: call {{.*}}<a, x, y>
        pass
