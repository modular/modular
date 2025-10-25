# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from simple_struct_package import MyStruct


trait MyTrait:
    fn get_value(self) -> Int:
        ...


__extension MyStruct(MyTrait):
    fn get_value(self) -> Int:
        return self.value
