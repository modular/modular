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
    def method(self) -> Int:
        ...


struct SomeStruct[param: Int](Trait, TrivialRegisterPassable):
    def method(self) -> Int:
        pass


def param_func[T: Trait](value: T) -> Int:
    pass


# CHECK-LABEL: lit.fn @"top
def top[pvalue: SomeStruct[2]]():
    # CHECK: lit.alias.decl [[alias_decl:.*]]: !lit.struct<#SomeStruct <:!Int {2}>> = <pvalue>
    comptime alias_decl = pvalue
    # CHECK: result{{.*}} = <apply{{.*}} store_to_mem(#alias_alias_decl))>
    comptime result = param_func(alias_decl)
