# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


def id_simd[w: Int](v: SIMD[DType.uint32, w]) -> SIMD[DType.uint32, w]:
    return v


def int32_simd[w: Int](v: SIMD[DType.int32, w]) -> SIMD[DType.int32, w]:
    return v


# expected-note @+1 {{function declared here}}
def take_closure_param[
    C: def[w: Int](v: SIMD[DType.uint32, w]) unified -> SIMD[DType.uint32, w]
]():
    pass


# expected-note @+1 {{function declared here}}
def take_closure_arg[
    F: def[w: Int](v: SIMD[DType.uint32, w]) unified -> SIMD[DType.uint32, w]
](f: F):
    pass


def test_function_to_wrapper_struct():
    # success: function-typed value -> concrete closure wrapper struct type.
    take_closure_arg(id_simd)

    # error: function-typed value with incompatible signature.
    # expected-error @below {{invalid call to 'take_closure_arg'}}
    take_closure_arg(int32_simd)


def test_wrapper_to_closure_param():
    def wrapped_ok[
        w: Int
    ](v: SIMD[DType.uint32, w]) unified {var} -> SIMD[DType.uint32, w]:
        return v

    def wrapped_bad[
        w: Int
    ](v: SIMD[DType.int32, w]) unified {var} -> SIMD[DType.int32, w]:
        return v

    # success: closure-wrapper value -> closure-typed parameter.
    take_closure_param[type_of(wrapped_ok)]()

    # error: closure-wrapper value with incompatible signature.
    # expected-error @below {{invalid call to 'take_closure_param'}}
    take_closure_param[type_of(wrapped_bad)]()


def test_wrapper_to_wrapper_struct():
    def wrapped_1[
        w: Int
    ](v: SIMD[DType.uint32, w]) unified {var} -> SIMD[DType.uint32, w]:
        return v

    def wrapped_2[
        w: Int
    ](v: SIMD[DType.int32, w]) unified {var} -> SIMD[DType.int32, w]:
        return v

    # success: closure-wrapper value -> same concrete wrapper struct type.
    take_closure_arg[type_of(wrapped_1)](wrapped_1)

    # error: closure-wrapper value with incompatible signature.
    # expected-error @below {{invalid call to 'take_closure_arg'}}
    take_closure_arg[type_of(wrapped_1)](wrapped_2)


def symbol_renamed[n: Int](x: SIMD[DType.uint32, n]) -> SIMD[DType.uint32, n]:
    return x


def test_canonical_equiv_symbol_to_parameter_success():
    def wrapped_renamed[
        n: Int
    ](x: SIMD[DType.uint32, n]) unified {var} -> SIMD[DType.uint32, n]:
        return x

    # Success: closure-struct type -> parameter type implicit conversion
    take_closure_param[type_of(wrapped_renamed)]()
