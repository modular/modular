# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


struct StringNoCopy:
    var size: __mlir_type.index

    fn __init__(out self):
        pass

    fn __del__(owned self):
        pass


fn makes_escaping_closurenocopy(m: StringNoCopy):
    # expected-error @below {{cannot synthesize memberwise init because field 'field0' has non-copyable and non-movable type 'StringNoCopy'}}
    fn myclosure() -> StringNoCopy:
        # expected-error @below {{'StringNoCopy' is not copyable because it has no '__copyinit__'}}
        return m

    var y: fn () escaping -> None = myclosure


# COM: https://github.com/modular/mojo/issues/1223
# COM: When a runtime argument has incorrect type, nested function bodies may
# COM: still be resolved. Ensure that we don't crash when the arg is used.
@value
struct Parametric[a: Index]:
    pass


fn test_suppressed_dyn_binding_error[
    x: Index
    # expected-error @below {{parametric functions may not be used as arguments; consider passing as a parameter instead}}
](pval: Parametric[x], func: fn[y: Index] (p: Parametric[y]) -> None):
    fn nested():
        func(pval)
