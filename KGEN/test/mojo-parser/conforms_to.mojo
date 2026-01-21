# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Foo:
    pass


trait Bar:
    pass


comptime Alias = Bar
# CHECK:      lit.alias.decl *"CONFORMS_TO_CHECK
# CHECK-SAME: conforms_to(:!mt_Foo !Foo, ["conforms_to::Bar"])
comptime CONFORMS_TO_CHECK = conforms_to(Foo, Alias)
