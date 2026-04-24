# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct PlainStruct:
    pass


def identity[T: Movable](var data: T) -> T:
    return data^


trait PackageTrait:
    def method(self):
        pass


trait PackageTrait2:
    def method2(self):
        pass


trait PackageChildTrait(PackageTrait, PackageTrait2):
    pass


trait UsedInPackageTrait:
    def method(self):
        pass


struct UseTrait(UsedInPackageTrait):
    def method(self):
        pass


struct UseTraitReg(RegisterPassable, UsedInPackageTrait):
    def method(self):
        pass


def trait_method[T: UsedInPackageTrait]():
    pass


# COM: Create a thunk that is only referenced in this module.
struct _PrivateReg(TrivialRegisterPassable, UsedInPackageTrait):
    def method(self):
        pass


@always_inline
def contains_thunk_ref():
    trait_method[_PrivateReg]()


# To test that linking multiple packages together works as expected, we wish to
# prevent this function definition from being inlined into modules that import
# it. Its definition should remain in this package module, and be callable from
# other modules.
@no_inline
def dont_inline_me():
    print("Don't you dare!")


@export
def exported_func():
    pass


# This function calls a closure, whose trait is preserved by the package. When
# a module using this package uses a closure with an identical signature, the
# declaration will first be pulled from this package. Thus the signatures of
# the two closures must match.
def call_closure[
    func: def[x: Int, y: Int, z: Int](idx: Int) -> None,
    //,
    simd_width: Int,
](size: Int, closure: func):
    closure[simd_width, 0, 1](0)
