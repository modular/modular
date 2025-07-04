# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn identity[T: Movable](var data: T) -> T:
    return data^


trait PackageTrait:
    fn method(self):
        pass


trait PackageTrait2:
    fn method2(self):
        pass


trait PackageChildTrait(PackageTrait, PackageTrait2):
    pass


trait UsedInPackageTrait:
    fn method(self):
        pass


struct UseTrait(UsedInPackageTrait):
    fn method(self):
        pass


@register_passable
struct UseTraitReg(UsedInPackageTrait):
    fn method(self):
        pass


fn trait_method[T: UsedInPackageTrait]():
    pass


# COM: Create a thunk that is only referenced in this module.
@register_passable("trivial")
struct _PrivateReg(UsedInPackageTrait):
    fn method(self):
        pass


@always_inline
fn contains_thunk_ref():
    trait_method[_PrivateReg]()


# To test that linking multiple packages together works as expected, we wish to
# prevent this function definition from being inlined into modules that import
# it. Its definition should remain in this package module, and be callable from
# other modules.
@no_inline
fn dont_inline_me():
    print("Don't you dare!")


@export
fn exported_func():
    pass
