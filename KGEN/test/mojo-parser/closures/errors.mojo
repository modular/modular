# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


struct StringNoCopy:
    var size: __mlir_type.index

    def __init__(out self):
        pass

    def __del__(deinit self):
        pass


def makes_escaping_closurenocopy(m: StringNoCopy):
    # expected-error @below {{cannot synthesize fieldwise init because field 'field0' has non-copyable and non-movable type 'StringNoCopy'}}
    def myclosure() -> StringNoCopy:
        # expected-error @below {{value of type 'StringNoCopy' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
        return m

    var y: def () escaping -> None = myclosure


# COM: https://github.com/modular/mojo/issues/1223
# COM: When a runtime argument has incorrect type, nested function bodies may
# COM: still be resolved. Ensure that we don't crash when the arg is used.
struct Parametric[a: Int]:
    pass


def test_suppressed_dyn_binding_error[
    x: Int
    # expected-error @below {{parametric functions may not be used as arguments; consider passing as a parameter instead}}
](pval: Parametric[x], func: def[y: Int] (p: Parametric[y]) -> None):
    def nested():
        func(pval)
