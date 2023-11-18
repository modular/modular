# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# COM: Validate parameters on a cache miss and check cache hit as well.
# RUN: kgen-translate -import-mojo --mojo-disable-parser-caching=true --mojo-disable-builtins %s | kgen-opt -verify-parameters | FileCheck %s
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


# CHECK: result = <apply{{.*}}store_to_mem(rebind(:{{.*}}@SomeStruct<2> : metatype<{{.*}}> {{.*}}alias_decl))
fn top[pvalue: SomeStruct[__mlir_attr.`2:index`]]():
    alias alias_decl = pvalue
    alias result = param_func(alias_decl)
