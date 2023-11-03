# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo -verify-diagnostics %s --mojo-disable-builtins


# expected-error @below {{value of type 'StringNoCopy' cannot be copied into its destination}}
struct StringNoCopy:
    var size: __mlir_type.index

    fn __init__(inout self):
        pass

    fn __del__(owned self):
        pass


fn makes_escaping_closurenocopy(m: StringNoCopy):
    fn myclosure() escaping -> StringNoCopy:
        # expected-error @below {{value of type 'StringNoCopy' cannot be copied into its destination}}
        return m
