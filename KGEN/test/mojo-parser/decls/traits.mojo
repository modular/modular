# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index
alias AnyRegType = __mlir_type.`!kgen.anyregtype`
alias StringLiteral = __mlir_type.`!kgen.string`

alias `1` = __mlir_attr.`1 : index`

struct object: pass
struct Error: pass

trait AnyType:
    fn __del__(owned self, /): ...

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.trait.decl @Trait<?, MT: regtype, T: !kgen.paramref<MT>>
trait Trait:
    # CHECK: lit.func @"f0(T)"[{{.*}}](%self: !lit.ref<:!kgen.paramref<MT> T, {{.*}}> borrow_in_mem) -> !kgen.none
    # CHECK-NEXT: lit.trait_func
    fn f0(self: Self): ...

    # CHECK: lit.func @"f1(T&)"{{.*}}(%self: !lit.ref<mut :!kgen.paramref<MT> T, {{.*}}> byref) -> !kgen.none
    # CHECK-NEXT: lit.trait_func
    fn f1(inout self: Self): ...

    # CHECK: lit.func @"f2(T&)"{{.*}}(%self: !lit.ref<mut :!kgen.paramref<MT> T, {{.*}}> byref) -> !kgen.none attributes
    # CHECK-NEXT: lit.trait_func
    fn f2(inout self: Self):
        pass

    # CHECK: lit.func @"f3(,T)"[{{.*}}](%__result__: !lit.ref<mut !object, {{.*}}> byref_result, |, %self: !lit.ref<mut :!kgen.paramref<MT> T, {{.*}}> owned_in_mem) throws -> !kgen.variant<!Error, none>
    # CHECK-NEXT: lit.trait_func
    def f3(self: Self):
        pass

    # CHECK: lit.func @"f4(,T&)"[{{.*}}](%__result__: !lit.ref<mut !object, {{.*}}> byref_result, |, %self: !lit.ref<mut :!kgen.paramref<MT> T, {{.*}}> byref) throws -> !kgen.variant<!Error, none>
    # CHECK-NEXT: lit.trait_func
    def f4(inout self: Self):
        pass

    fn overloaded(self): ...
    fn overloaded(self, x: Int): ...
    fn overloaded(self, x: StringLiteral): ...

    # CHECK-LABEL: lit.func @"parametric
    # CHECK-SAME: <[[x:.*]][x]>
    fn parametric[x: Int](self): ...

# CHECK-LABEL: lit.trait.decl @EmptyTrait<?, MT: regtype, T: !kgen.paramref<MT>>
trait EmptyTrait:
    pass

# CHECK-LABEL: lit.trait.decl @Trait1<?, MT: regtype, T: !kgen.paramref<MT>>
trait Trait1:
    # CHECK: lit.func @"f{{.*}}(%__result__: !lit.ref<mut :!kgen.paramref<MT> T, {{.*}}> byref_result, |, %self: !lit.ref<:!kgen.paramref<MT> T, {{.*}}> borrow_in_mem) -> !kgen.none
    fn f(self: Self) -> Self: ...

# CHECK-LABEL: lit.trait.decl @Trait2<?, MT: regtype, T: !kgen.paramref<MT>>
trait Trait2:
    # CHECK: lit.func @"f{{.*}}(%__result__: !lit.ref<mut :!kgen.paramref<MT> T, {{.*}}> byref_result, |, %self: !lit.ref<:!kgen.paramref<MT> T, {{.*}}> borrow_in_mem) -> !kgen.none
    fn f(self: Self) -> Self: ...

# CHECK-LABEL: lit.struct.decl @StructWithTraits(trait<{{.*}}@Trait1>, trait<{{.*}}@Trait2>)
struct StructWithTraits(Trait1, Trait2):
    # CHECK: lit.func @"f{{.*}}(%{{.*}}: !lit.ref<mut !StructWithTraits, {{.*}}> byref_result, |, %self: !lit.ref<!StructWithTraits, {{.*}}> borrow_in_mem) -> !kgen.none
    fn f(self: Self) -> Self: ...

# CHECK-LABEL: lit.trait.decl @CFMTrait<?, MT: regtype, T: !kgen.paramref<MT>>
trait CFMTrait:
   #CHECK: lit.func @"f1(T)"[{{.*}}](%self: !lit.ref<:!kgen.paramref<MT> T, {{.*}}> borrow_in_mem) -> !kgen.none
   fn f1(self: Self):
        pass

   #CHECK: lit.func @"f2()"() -> !kgen.none
   @staticmethod
   fn f2():
       pass

# CHECK-LABEL: lit.struct.decl @CFMStruct(trait<{{.*}}@CFMTrait>
struct CFMStruct(CFMTrait):
   #CHECK: lit.func @"f1({{.*}})"[{{.*}}](%self: !lit.ref<!CFMStruct, {{.*}}> borrow_in_mem) -> !kgen.none
   fn f1(self: Self):
       pass

   #CHECK: lit.func @"f2()"() -> !kgen.none
   @staticmethod
   fn f2():
       pass

# Test for struct with parameters and function with parameters.
# CHECK-LABEL: lit.trait.decl @CFMTraitParams<?, MT: regtype, T: !kgen.paramref<MT>>
trait CFMTraitParams:
    # CHECK: lit.func @"f1{{.*}}"[{{.*}}]<[[TT:_.*]]: trait<[[TN:@.*]]>>(%self: !lit.ref<:!kgen.paramref<MT> T, {{.*}}> borrow_in_mem)
    fn f1[x: CFMTraitParams](self):
        pass

# CHECK-LABEL: lit.struct.decl @CFMStructParams
struct CFMStructParams[t1: AnyRegType, t2: AnyRegType](CFMTraitParams):
    # CHECK: lit.func @"f1{{.*}}"[{{.*}}]<[[ST1:_.*]]: trait<[[TN:@.*]]>>(%self: !lit.ref<{{.*}}@CFMStructParams<:regtype [[T1:_.*]], :regtype [[T2:_.*]]>{{.*}}> borrow_in_mem)
    fn f1[x: CFMTraitParams](self):
       pass

# CHECK-LABEL: lit.func @"generic_trait_fn
# CHECK-SAME: <[[T:.*_T]][T]: trait<{{.*}}@Trait>>
# CHECK-SAME: %x: !lit.ref<:trait<{{.*}}@Trait> [[T]], {{.*}}> borrow_in_mem
fn generic_trait_fn[T: Trait](x: T):
    # CHECK: call_param[!lit.signature<[1]("self": {{.*}} borrow_in_mem) -> !kgen.none>:
    # CHECK-SAME: get_type_method(:trait<{{.*}}@Trait> [[T]], "f0")]{{.*}}(%x)
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
    # CHECK-SAME: get_type_method(:{{.*}} [[T]], "parametric"), 1)
    x.parametric[`1`]()

# CHECK-LABEL: lit.func @"existential_arg
# CHECK-SAME: (%x: !lit.ref<trait<{{.*}}@Trait>, {{.*}}>
fn existential_arg(x: Trait):
    pass



trait SimpleTrait:
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

# CHECK-LABEL: lit.func @"test_metatype_to_trait_vtable
fn test_metatype_to_trait_vtable():
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:trait<{{.*}}@SimpleTrait> [!TraitStruct{{[0-9]*}}, {
    # CHECK-SAME: "method" : !lit.signature<[1]("self": !lit.ref<!TraitStruct, {{.*}}> borrow_in_mem, "y": index borrow) -> !kgen.none> = {{.*}}@TraitStruct::@"method
    # CHECK-SAME: "param_method" : !lit.signature<[1]<"x": index>("self": !lit.ref<!TraitStruct, {{.*}}> borrow_in_mem) -> !kgen.none> = {{.*}}@TraitStruct::@"param_method{{.*}}"<?>
    take_simple_trait[TraitStruct]()
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:trait<{{.*}}@SimpleTrait> [{{.*}}@ParametricTraitStruct<2> : metatype<{{.*}}>, {
    # CHECK-SAME: "method" : !lit.signature<[1]("self": {{.*}}@ParametricTraitStruct<2>{{.*}} borrow_in_mem, "y": index borrow) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"method{{.*}}"<2>,
    # CHECK-SAME: "param_method" : !lit.signature<[1]<"x": index>("self": {{.*}}@ParametricTraitStruct<2>{{.*}} borrow_in_mem) -> !kgen.none> = {{.*}}@ParametricTraitStruct::@"param_method{{.*}}"<2, ?>
    take_simple_trait[ParametricTraitStruct[__mlir_attr.`2 : index`]]()

# CHECK-LABEL: lit.func @"test_infer_trait
fn test_infer_trait(a: TraitStruct, b: ParametricTraitStruct[__mlir_attr.`2 : index`]):
    # CHECK: call {{.*}}infer_trait{{.*}}<:trait<{{.*}}@SimpleTrait> [!TraitStruct,
    infer_trait(a)
    # CHECK: call {{.*}}infer_trait{{.*}}<:trait<{{.*}}@SimpleTrait> [{{.*}}@ParametricTraitStruct<2> : metatype<{{.*}}>,
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

# CHECK-LABEL: lit.func @"trait_static_method
# CHECK-SAME: <[[T:.*]][T]
fn trait_static_method[T: StaticMethodTrait]():
    # CHECK: call_param[!lit.signature<() -> !kgen.none>: get_type_method(:trait<{{.*}}@StaticMethodTrait> [[T]], "foobar")]()
    T.foobar()

# CHECK-LABEL: lit.func @"copy_me
# CHECK-SAME: <[[T:.*]][T]
# CHECK-SAME: %__result__: !lit.ref<mut :trait<{{.*}}@Copyable> [[T]], {{.*}}> byref_result, |,
# CHECK-SAME: %value: !lit.ref<:trait<{{.*}}@Copyable> [[T]], {{.*}}> borrow_in_mem)
fn copy_me[T: Copyable](value: T) -> T:
    # CHECK-NEXT: call_param[!lit.signature<[2]("self": {{.*}}[[T]], {{.*}}> init_self, "existing": {{.*}}[[T]], {{.*}}> borrow_in_mem, |) -> !kgen.none>:
    # CHECK-SAME: get_type_method({{.*}} [[T]], "__copyinit__")]{{.*}}(%__result__, %value)
    return value

# CHECK-LABEL: lit.func @"move_me
# CHECK-SAME: <[[T:.*]][T]
# CHECK-SAME: @Movable> [[T]], {{.*}}> byref_result
# CHECK-SAME: @Movable> [[T]], {{.*}}> owned_in_mem
fn move_me[T: Movable](owned value: T) -> T:
    # CHECK-NEXT: %0 = lit.ownership.end_lifetime %value
    # CHECK-NEXT: call_param[{{.*}}get_type_method({{.*}} [[T]], "__moveinit__")]{{.*}}(%__result__, %0)
    return value ^

# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    fn __init__(inout self, x: Int):
        ...

    fn __copyinit__(inout self, existing: Self):
        ...

    @staticmethod
    fn may_throw() raises -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @RegTraitType
@register_passable
struct RegTraitType(TraitForReg):
    # CHECK-LABEL: lit.func @"`thunk___init__
    # CHECK-SAME: %self: !lit.ref<mut !RegTraitType, {{.*}}> init_self, |, %x: index borrow) -> !kgen.none
    fn __init__(x: Int) -> Self:
        # CHECK: %0 = lit.call {{.*}}@RegTraitType{{.*}}__init__{{.*}}(%x)
        # CHECK: store %0, %self
        pass

    # CHECK-LABEL: lit.func @"`thunk___copyinit__
    # CHECK-SAME: %self: !lit.ref<mut !RegTraitType, {{.*}}> init_self, |, %arg[existing]: !lit.ref<!RegTraitType, {{.*}}> borrow_in_mem) -> !kgen.none
    fn __copyinit__(existing: Self) -> Self:
        # CHECK: %0 = lit.ref.load %arg
        # CHECK: %1 = lit.call {{.*}}@RegTraitType{{.*}}__copyinit__{{.*}}(%0)
        # CHECK: store %1, %self
        pass

    # CHECK-LABEL: lit.func @"`thunk_may_throw
    # CHECK-SAME: %__result__: !lit.ref<mut !RegTraitType, {{.*}}> byref_result
    # CHECK-SAME: throws -> !kgen.variant<!Error, none> always_inline
    @staticmethod
    fn may_throw() raises -> Self:
        # CHECK: %0 = lit.call {{.*}}@RegTraitType::@"may_throw()"
        # CHECK: %1 = lit.handle_variant %0
        # CHECK: store %1, %__result__
        pass


trait CrazyTrait:
    pass

    fn foo[b: Int->d: Int](self, c: Int) -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @CrazyRegisterPassable
# CHECK-SAME: <[[a:.*]][a]>
@value
@register_passable
struct CrazyRegisterPassable[a: Int](CrazyTrait):
    pass

    # CHECK-LABEL: lit.func @"`thunk_foo
    # CHECK-SAME: <b[b] -> o0>(%__result__: !lit.ref<mut {{.*}}@CrazyRegisterPassable<[[a]]>>{{.*}} byref_result, |,
    # CHECK-SAME: %self: !lit.ref<{{.*}}@CrazyRegisterPassable<[[a]]>{{.*}} borrow_in_mem
    # CHECK-SAME: %c: index borrow) -> !kgen.none
    fn foo[b: Int->d: Int](self, c: Int) -> Self:
        # CHECK: %0 = lit.ref.load %self
        # CHECK: %1 = lit.call {{.*}}@CrazyRegisterPassable::@"foo{{.*}}<[[a]], b -> r0>(%0, %c)
        # CHECK: lit.ref.store %1, %__result__
        # CHECK: lit.param_return<r0>
        param_return[__mlir_attr.`2:index`]
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

    # CHECK: call_param{{.*}}@ChangedResultTypeStruct::@"`thunk_result_type()"
    convert_result_type[ChangedResultTypeStruct]()


trait SimpleTraitMethod:
    fn foo(self):
        ...

@register_passable
struct VariadicTrait[*I: Int](SimpleTraitMethod):
    fn foo(self):
        pass


# CHECK-LABEL: lit.func @"test_bind_variadic
fn test_bind_variadic():
    @parameter
    fn bind_trait[T: SimpleTraitMethod]():
        pass

    # CHECK: call_param
    # CHECK: "foo" : !lit.signature<[1]("self": {{.*}}<:variadic<index> []>{{.*}} borrow_in_mem) -> !kgen.none> = {{.*}}`thunk_foo{{.*}}"<:variadic<index> []>
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
    # CHECK-LABEL: lit.func @"`thunk_take
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
    # CHECK: lit.varlet.decl "c" let : !lit.ref<mut !NoDtor,
    # CHECK-NEXT: call {{.*}}default_construct{{.*}}<:trait<{{.*}}> [!NoDtor,{{.*}}(%c)
    let c = default_construct[NoDtor]()
    # CHECK: call {{.*}}@NoDtor::@"method
    c.method()

# CHECK-LABEL: lit.struct.decl @RegTrivialSpecial
@register_passable("trivial")
struct RegTrivialSpecial(AnyType, Copyable, Movable):
    pass
    # CHECK: lit.func @"`thunk___del__
    # CHECK-SAME: %0[{{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug
    # CHECK: return %none

    # CHECK: lit.func @"`thunk___copyinit__
    # CHECK-SAME: %0[{{.*}} init_self, %1[{{.*}} borrow_in_mem
    # CHECK-NEXT: [[V:%.*]] = lit.ref.load %1
    # CHECK-NEXT: lit.ref.store [[V]], %0

    # CHECK: lit.func @"`thunk___moveinit__{{.*}}%0[{{.*}} init_self, %1[{{.*}} owned_in_mem
    # CHECK: [[V:%.*]] = lit.load.consume
    # CHECK-NEXT: lit.ref.store [[V]], %0

# CHECK-LABEL: lit.struct.decl @RegSpecial
@register_passable
struct RegSpecial(AnyType, Copyable, Movable):
    fn __copyinit__(existing: Self) -> Self:
        return Self {}

    # CHECK: lit.func @"`thunk___del__
    # CHECK-SAME: %0[{{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug
    # CHECK: return %none

    # CHECK: lit.func @"`thunk___moveinit__
    # CHECK-SAME: %0[{{.*}} init_self, %1[{{.*}} owned_in_mem
    # CHECK-NEXT: [[V:%.*]] = lit.load.consume %1
    # CHECK-NEXT: lit.ref.store [[V]], %0

# CHECK-LABEL: lit.struct.decl @MemoryOnlySpecial
struct MemoryOnlySpecial(AnyType, Copyable, Movable):
    fn __copyinit__(inout self, existing: Self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    # CHECK: lit.func @"`thunk___del__
    # CHECK-SAME: %0[{{.*}} owned_in_mem, |) -> !kgen.none always_inline_no_debug
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
    fn foo(self): ...

# CHECK-LABEL: lit.trait.decl @ChildTraitSameSig
trait ChildTraitSameSig(ParentTraitSameSig):
    # CHECK-NEXT: lit.func @"foo(T)"
    # CHECK-NEXT: lit.trait_func
    fn foo(self): ...
    # CHECK-NOT: foo

# CHECK-LABEL: lit.trait.decl @GreatGrandFather
# CHECK-SAME: (trait<{{.*}}@AnyType>)
trait GreatGrandFather:
    # CHECK: lit.func @"foo(T)"
    fn foo(self): ...

# CHECK-LABEL: lit.trait.decl @GrandFather
# CHECK-SAME: (trait<{{.*}}@GreatGrandFather>,
trait GrandFather(GreatGrandFather):
    # CHECK: lit.func @"bar(T)"
    fn bar(self): ...
    # CHECK: lit.func @"foo(T)"

# CHECK-LABEL: lit.trait.decl @Father
# CHECK-SAME: (trait<{{.*}}@GrandFather>, trait<{{.*}}@GreatGrandFather>[trait<{{.*}}@GrandFather>],
trait Father(GrandFather):
    # CHECK: lit.func @"baz(T)"
    fn baz(self): ...
    # CHECK: lit.func @"bar(T)"
    # CHECK: lit.func @"foo(T)"

# CHECK-LABEL: lit.struct.decl @TraitInheritance
# CHECK-SAME: (trait<{{.*}}@Father>, trait<{{.*}}@GrandFather>[trait<{{.*}}@Father>], trait<{{.*}}@GreatGrandFather>[trait<{{.*}}@GrandFather>, trait<{{.*}}@Father>],
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
# CHECK-SAME: <[[T:.*]][T]: trait<{{.*}}@Father>>
fn pass_up_trait[T: Father](x: T):
    # CHECK-NEXT: call {{.*}}infer_grand_father{{.*}}<:trait<{{.*}}@GrandFather>
    # CHECK-SAME: [!kgen.paramref<:trait<{{.*}}@Father> [[T]]>, {
    # CHECK-SAME: "bar" : !lit.signature<[1]("self": !lit.ref<:trait<{{.*}}@Father> [[T]], {{.*}}> borrow_in_mem) -> !kgen.none> = get_type_method({{.*}} [[T]], "bar"),
    # CHECK-SAME: "foo" : !lit.signature<[1]("self": !lit.ref<:trait<{{.*}}@Father> [[T]], {{.*}}> borrow_in_mem) -> !kgen.none> = get_type_method({{.*}} [[T]], "foo")
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
    # CHECK: call {{.*}}take_movable{{.*}}"__moveinit__" : {{.*}} = rebind({{.*}}`thunk___moveinit__
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
    # CHECK: lit.func @"__init__
    # CHECK: lit.func @"`thunk___del__

# CHECK-LABEL: lit.struct.decl @HasMyPointerSelf
struct HasMyPointerSelf(AnyType):
    # CHECK: lit.struct.field x : !kgen.declref<{{.*}}@MyPointer<:trait<{{.*}}@AnyType>
    var x: MyPointer[Self]
    # CHECK: lit.func @"`thunk___del__

    fn __moveinit__(inout self, owned existing: Self, /):
        pass

# Parser crash
# https://github.com/modularml/modular/issues/27897
# CHECK-LABEL: lit.func @"check_trait_conversion_bymem_result_alias_crash
fn retMemory[T: TraitForReg](value: T) -> MemoryOnlySpecial: pass
fn check_trait_conversion_bymem_result_alias_crash(x: RegTraitType) -> MemoryOnlySpecial:
   return retMemory(x)


# Calling functions with implicit lifetimes needs to cooperate.
fn test[a: ABC]():
   _ = ABCOptionalParamInt[ABCDim(a)]()

trait SomeTrait: pass

struct ABC(SomeTrait):
  fn __init__(inout self) : pass

@register_passable("trivial")
struct ABCOptionalParamInt[dim_parametric: ABCDim]:
    fn __init__() -> Self:
        pass

struct ABCDim:
  fn __init__[type: SomeTrait](inout self, value: type):
      pass
