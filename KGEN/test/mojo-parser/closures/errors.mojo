# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo -verify-diagnostics %s --mojo-disable-builtins


# expected-error @below {{'StringNoCopy' is not copyable because it has no '__copyinit__'}}
# expected-error @below {{'StringNoCopy' is not copyable or movable because it has no '__copyinit__' or '__moveinit__' member}}
struct StringNoCopy:
    var size: __mlir_type.index

    fn __init__(inout self):
        pass

    fn __del__(owned self):
        pass


fn makes_escaping_closurenocopy(m: StringNoCopy):
    fn myclosure() escaping -> StringNoCopy:
        # expected-error @below {{'StringNoCopy' is not copyable because it has no '__copyinit__'}}
        return m

    # expected-error @below {{cannot implicitly convert 'fn() escaping -> StringNoCopy' value to 'fn() escaping -> None' in 'var' initializer}}
    var y: fn () escaping -> None = myclosure
