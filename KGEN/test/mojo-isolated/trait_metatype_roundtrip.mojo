# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Validate parameters on a cache miss and check cache hit as well.
# RUN: %parse-mojo-isolated %s | FileCheck %s
# RUN: %parse-mojo-isolated %s -o /dev/null -bytecode-output - | kgen-opt | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


trait Trait:
    fn method(self) -> Int:
        pass


@register_passable("trivial")
struct SomeStruct[param: Int](Trait):
    fn method(self) -> Int:
        pass


fn param_func[T: Trait](value: T) -> Int:
    pass


# CHECK-LABEL: lit.fn @"top
fn top[pvalue: SomeStruct[2]]():
    # CHECK: alias.decl [[alias_decl:.*]]: @{{.*}} = <pvalue>
    alias alias_decl = pvalue
    # CHECK: result{{.*}} = <apply{{.*}} store_to_mem(sugar_alias(*"alias_decl`", pvalue)))>
    alias result = param_func(alias_decl)
