# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics -o /dev/null


# We should be able to infer something with a concrete origin even if it
# requires an upcast.
def takeA[origin: Origin, T: AnyType](ref[origin] a: String, b: T):
    pass


def infer_ref_argument():
    var s: String
    var t: String
    takeA(s, t)  # Ok.
    takeA[AnyOrigin[mut=True]](s, t)  # Ok.


@fieldwise_init
struct TwoIntParamStruct[a: Int, b: Int]:
    pass


# expected-note @below {{function declared here}}
def take_two_int_dep[x: Int](a: TwoIntParamStruct[x, x + 1]):
    pass


# expected-note @+1 {{function declared here}}
def take_tied_values[
    first1: Int, first2: Int, second: Int
](a: TwoIntParamStruct[first1, second], b: TwoIntParamStruct[first2, second],):
    pass


def infer_two_param_dep_struct[y: Int]():
    take_two_int_dep(TwoIntParamStruct[1, 2]())
    # expected-error @+1 {{value passed to 'a' cannot be converted from 'TwoIntParamStruct[2, 2]' to 'TwoIntParamStruct[2, 3]'}}
    take_two_int_dep(TwoIntParamStruct[2, 2]())
    take_two_int_dep(TwoIntParamStruct[b=2, a=1]())

    take_two_int_dep(TwoIntParamStruct[y, y + 1]())
    # expected-error @+1 {{value passed to 'a' cannot be converted from 'TwoIntParamStruct[y, (y + 2)]' to 'TwoIntParamStruct[y, (y + 1)]'}}
    take_two_int_dep(TwoIntParamStruct[y, y + 2]())

    # expected-error @+1 {{value passed to 'b' cannot be converted from 'TwoIntParamStruct[20, 2]' to 'TwoIntParamStruct[20, 1]'}}
    take_tied_values(TwoIntParamStruct[10, 1](), TwoIntParamStruct[20, 2]())
