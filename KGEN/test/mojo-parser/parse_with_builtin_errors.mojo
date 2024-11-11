# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s


# expected-note @below {{'Error' declared here}}
fn Error():
    pass


struct FailingStruct:
    # expected-error @below {{'Error' doesn't resolve to a type}}
    fn failure(owned self) raises:
        pass

# MOCO-846: bad message when types don't match due to parameter expressions
# that can't be evaluated at overload resolution time.
struct HasSize[size: Int]:
    fn __init__(out self: Self):
        pass

# expected-note @below {{function declared here}}
fn has_expr_for_elaborator[width: Int](x: HasSize[width + 4]):
    pass

fn use_take_args():
    alias width = 4
    # expected-error @below {{cannot be converted}}
    # expected-note @below {{types parameters include unfolded expression at parser time; try rebinding to a consistent type?}}
    _ = has_expr_for_elaborator[width](HasSize[size=width + 4]())
