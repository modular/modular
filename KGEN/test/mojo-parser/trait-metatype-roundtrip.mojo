# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# COM: Validate parameters on a cache miss and check cache hit as well.
# RUN: kgen-translate -import-mojo --mojo-disable-parser-caching=true --mojo-disable-builtins %s | kgen-opt -verify-parameters | FileCheck %s
# RUN: kgen-translate -import-mojo --mojo-disable-parser-caching=true --mojo-disable-builtins %s -o /dev/null -bytecode-output - | kgen-opt -verify-parameters | FileCheck %s
# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | kgen-opt -verify-parameters | FileCheck %s
# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | kgen-opt -verify-parameters | FileCheck %s

alias int = __mlir_type.index


trait Trait:
    fn method(self) -> int:
        pass


@register_passable("trivial")
struct SomeStruct[param: int](Trait):
    fn method(self) -> int:
        pass


fn param_func[T: Trait](value: T) -> int:
    pass


# CHECK-LABEL: lit.func @"top
fn top[pvalue: SomeStruct[__mlir_attr.`2:index`]]():
    # CHECK: alias.decl [[alias_decl:.*]]: @
    alias alias_decl = pvalue
    # CHECK: result = <apply{{.*}}store_to_mem([[alias_decl]]))
    alias result = param_func(alias_decl)
