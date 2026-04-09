# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: %parse-mojo-isolated %s | FileCheck %s


@export("my_named_export", ABI="C")
# CHECK: lit.fn export C @"export_me()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_named_export" : !kgen.string, false>
def export_me() raises -> None:
    ...


@export
# CHECK: lit.fn export @"not_c_exported()"
def not_c_exported():
    pass


struct Thing:
    # CHECK: lit.fn export @"member
    @export
    def member(self):
        pass
