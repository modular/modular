# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Verify that untyped mlir typed trait defined aliases get type annotations generated for them.
# CHECK-DAG: #type_value = #kgen.type<index, {"__del__" : !lit.generator<[1]{{.*}}> = @{{.*}}"__del__{{.*}}"<:type index>}> : !AnyType

# Binding `t` to `MainTraitT` triggers the following type constants
# CHECK-DAG: #ImplT1 = #kgen.type<!ImplT, {"subget" : !lit.generator<("self": !ImplT) -> index> = @{{.*}}::@ImplT::@"subget{{.*}}", "__del__" : {{.*}}}> : !SubTraitT
# CHECK-DAG: #MainImplT2 = #kgen.type<!MainImplT, {"ret_type" : !SubTraitT = #ImplT1, "anything" : !AnyType = #type_value, "get" : !lit.generator<("self": !MainImplT) -> !ImplT> = @{{.*}}::@MainImplT::@"get{{.*}}", "__del__" : {{.*}}}> : !MainTraitT

# Binding `t` to `MainTraitT2` triggers the following type constants
# CHECK-DAG: #ImplT2 = #kgen.type<!ImplT, {"subget2" : !lit.generator<("self": !ImplT) -> index> = @{{.*}}::@ImplT::@"subget2{{.*}}", "__del__" : {{.*}}}> : !SubTraitT2
# CHECK-DAG: #MainImplT1 = #kgen.type<!MainImplT, {"ret_type" : !SubTraitT2 = #ImplT2, "get2" : !lit.generator<("self": !MainImplT) -> !ImplT> = @{{.*}}::@MainImplT::@"get{{.*}}", "__del__" : {{.*}}}> : !MainTraitT

@register_passable("trivial")
trait SubTraitT:
    fn subget(self) -> Index:
        ...

@register_passable("trivial")
trait SubTraitT2:
    fn subget2(self) -> Index:
        ...

@register_passable("trivial")
trait MainTraitT:
    alias ret_type: SubTraitT
    alias anything: AnyType
    fn get(self) -> Self.ret_type:
        ...

@register_passable("trivial")
trait MainTraitT2:
    alias ret_type: SubTraitT2
    fn get2(self) -> Self.ret_type:
        ...

@fieldwise_init
@register_passable("trivial")
struct ImplT(SubTraitT, SubTraitT2):
    fn subget(self) -> Index:
        return `0`

    fn subget2(self) -> Index:
        return `0`

    fn BAR(self) -> Index:
        return `1`

@fieldwise_init
@register_passable("trivial")
struct MainImplT(MainTraitT, MainTraitT2):
    # CHECK: lit.alias.decl *"ret_type{{.*}}": !mt_ImplT = <!ImplT>
    alias ret_type = ImplT
    # CHECK: lit.alias.decl *"anything{{.*}}": type = <index>
    alias anything = Index

    fn get(self) -> Self.ret_type:
        return ImplT()

    fn get2(self) -> Self.ret_type:
        return ImplT()

    fn doSomethingNonTraity(self) -> Index:
        # Verify the ImplT type is returned, not a type value of trait metatype.
        # CHECK: lit.call @{{.*}}::@MainImplT::@"get{{.*}}"(%self) : !lit.generator<("self": !MainImplT) -> !ImplT>
        var impl = self.get()
        var a = impl.BAR()
        return a

fn repro_issue[main_t: MainTraitT, main_t2: MainTraitT2](t: main_t, t2: main_t2) -> Index:
    var a = t.get().subget()
    var b = t2.get2().subget2()
    var c = __mlir_op.`index.add`(a, b)
    return c

@export
fn callIt() -> Index:
    var t = MainImplT()
    var a = repro_issue(t, t)
    return a

# ===----------------------------------------------------------------------=== #
# Upcast tests
# ===----------------------------------------------------------------------=== #

# Just make sure this parses.
fn declval[T: AnyType]() -> T: pass
trait MyThingTrait:
    fn thing(self) -> __mlir_type.i1: ...
fn propagate_type[T: MyThingTrait](range: T) -> __type_of(declval[T]().thing()):
    pass
