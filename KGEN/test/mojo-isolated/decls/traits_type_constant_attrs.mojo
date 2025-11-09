# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

@register_passable("trivial")
trait SubTraitT:
    fn subget(self) -> Int:
        ...


@register_passable("trivial")
trait SubTraitT2:
    fn subget2(self) -> Int:
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
    fn subget(self) -> Int:
        return 0

    fn subget2(self) -> Int:
        return 0

    fn BAR(self) -> Int:
        return 1


@fieldwise_init
@register_passable("trivial")
struct MainImplT(MainTraitT, MainTraitT2):
    # CHECK: lit.alias.decl *"ret_type{{.*}}": !mt_ImplT = <!ImplT>
    alias ret_type = ImplT
    # CHECK: lit.alias.decl *"anything{{.*}}": !mt_Int = <!Int>
    alias anything = Int

    fn get(self) -> Self.ret_type:
        return ImplT()

    fn get2(self) -> Self.ret_type:
        return ImplT()

    fn doSomethingNonTraity(self) -> Int:
        # Verify the ImplT type is returned, not a type value of trait metatype.
        # CHECK: lit.call @{{.*}}::@MainImplT::@"get{{.*}}"(%self) : !lit.generator<("self": !MainImplT)
        # CHECK-SAME: -> !ImplT
        var impl = self.get()
        var a = impl.BAR()
        return a


fn repro_issue[
    main_t: MainTraitT, main_t2: MainTraitT2
](t: main_t, t2: main_t2) -> Int:
    var a = t.get().subget()
    var b = t2.get2().subget2()
    var c = __mlir_op.`index.add`(a._mlir_value, b._mlir_value)
    return Int(mlir_value=c)


@export
fn callIt() -> Int:
    var t = MainImplT()
    var a = repro_issue(t, t)
    return a


# ===----------------------------------------------------------------------=== #
# Upcast tests
# ===----------------------------------------------------------------------=== #


# Just make sure this parses.
fn declval[T: AnyType]() -> T:
    pass


trait MyThingTrait:
    fn thing(self) -> __mlir_type.i1:
        ...


fn propagate_type[T: MyThingTrait](range: T) -> type_of(declval[T]().thing()):
    pass
