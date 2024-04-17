# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-error @below {{'StringNoCopy' is not copyable because it has no '__copyinit__'}}
# expected-error @below {{'StringNoCopy' is not copyable or movable because it has no '__copyinit__' or '__moveinit__' member}}
struct StringNoCopy:
    var size: __mlir_type.index

    fn __init__(inout self):
        pass

    fn __del__(owned self):
        pass


fn makes_escaping_closurenocopy(m: StringNoCopy):
    fn myclosure() -> StringNoCopy:
        # expected-error @below {{'StringNoCopy' is not copyable because it has no '__copyinit__'}}
        return m

    # expected-error @below {{cannot implicitly convert 'fn() escaping -> StringNoCopy' value to 'fn() escaping -> None' in 'var' initializer}}
    var y: fn () escaping -> None = myclosure


# COM: https://github.com/modularml/mojo/issues/1223
# COM: When a runtime argument has incorrect type, nested function bodies may
# COM: still be resolved. Ensure that we don't crash when the arg is used.
@value
struct Parametric[a: int]:
    pass


fn test_suppressed_dyn_binding_error[
    x: int
    # expected-error @below {{parametric functions may not be used as arguments; consider passing as a parameter instead}}
](pval: Parametric[x], func: fn[y: int] (p: Parametric[y]) -> None):
    fn nested():
        func(pval)
