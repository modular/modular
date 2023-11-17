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
