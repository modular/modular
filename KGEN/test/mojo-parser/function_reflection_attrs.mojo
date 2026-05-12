# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


comptime parameter_count_of[
    func_type: AnyType, //, func: func_type
]: Int = Int(
    mlir_value=__mlir_attr[
        `#kgen.get_function_parameter_count<`, func, `> : index`
    ]
)

comptime parameter_names_of[
    func_type: AnyType, //, func: func_type
] = ParameterList[
    __mlir_attr[
        `#kgen.get_function_parameter_names<`,
        func,
        `> : !kgen.param_list<!kgen.string>`,
    ]
]

comptime is_raising_of[
    func_type: AnyType, //, func: func_type
] = __mlir_attr[
    `#kgen.get_function_is_raising<`, func, `> : i1`
]


# ===----------------------------------------------------------------------=== #
# Concrete Function References — Parse-Time Folding
# ===----------------------------------------------------------------------=== #


def no_params():
    pass


def two_params[T: AnyType, n: Int](x: Int):
    pass


def no_raise_fn():
    pass


def raises_fn() raises:
    pass


# CHECK-LABEL: lit.fn @"main()"
def main():
    # parameter_count folds against a concrete `lit.fn` at parse time.
    # CHECK: lit.alias.decl *"countNoParams`{{[0-9]*}}": !Int = <{0}>
    comptime countNoParams = parameter_count_of[no_params]
    # CHECK: lit.alias.decl *"countTwoParams`{{[0-9]*}}": !Int = <{2}>
    comptime countTwoParams = parameter_count_of[two_params[Int, 4]]

    # parameter_names folds to a concrete `param_list<string>` at parse time.
    # CHECK: lit.alias.decl *"namesEmpty`{{[0-9]*}}": {{.*}}:param_list<string> []>
    comptime namesEmpty = parameter_names_of[no_params]
    # CHECK: lit.alias.decl *"namesTwo`{{[0-9]*}}": {{.*}}:param_list<string> ["T", "n"]>
    comptime namesTwo = parameter_names_of[two_params[Int, 4]]

    # is_raising folds at parse time to an i1 constant.
    # CHECK: lit.alias.decl *"raisingFalse`{{[0-9]*}}": i1 = <0>
    comptime raisingFalse = is_raising_of[no_raise_fn]
    # CHECK: lit.alias.decl *"raisingTrue`{{[0-9]*}}": i1 = <1>
    comptime raisingTrue = is_raising_of[raises_fn]


# ===----------------------------------------------------------------------=== #
# Non-Immediate Function Values — No Parse-Time Folding
# ===----------------------------------------------------------------------=== #


# When the function value is a generic parameter (a `param.decl.ref` rather
# than a `symbol.constant`), parser-time evaluation cannot resolve a defining
# op, so the reflection attribute persists symbolically. Folding only happens
# later, at the call site where the parameter is bound to a concrete function.

# CHECK-LABEL: lit.fn @"forwarded_count
def forwarded_count[func_type: AnyType, //, func: func_type]() -> Int:
    # CHECK: kgen.get_function_parameter_count<#kgen.param.decl.ref<"func">
    return parameter_count_of[func]


# CHECK-LABEL: lit.fn @"forwarded_raising
def forwarded_raising[func_type: AnyType, //, func: func_type]() -> Bool:
    # CHECK: kgen.get_function_is_raising<#kgen.param.decl.ref<"func">
    return Bool(is_raising_of[func])
