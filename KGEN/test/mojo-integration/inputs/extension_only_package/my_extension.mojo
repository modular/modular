# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from struct_only_package import MyStruct


__extension MyStruct:
    def get_speed(self) -> Int:
        return self.speed
