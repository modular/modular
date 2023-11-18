# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn identity[T: AnyRegType](data: T) -> T:
    return data


trait PackageTrait:
    fn method(self):
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
