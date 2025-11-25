# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .base import MyStruct


# User will import this directly, not the extension.
struct IntConfig:
    fn __init__(out self):
        pass


# This extension will hitch a ride on any imports of anything in this file
# (like IntConfig) to make itself known to anybody who deals with anyone in
# this file.
__extension MyStruct:
    fn intermediate_method(self):
        print("intermediate_method from extension")
