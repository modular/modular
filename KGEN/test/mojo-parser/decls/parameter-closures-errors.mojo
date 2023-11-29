# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

alias Int = __mlir_type.index

fn bind_fat_to_thin_target[g: fn(y: Int) -> Int](x: Int): pass

fn bind_fat_to_thin_main():
    let x = __mlir_attr.`4 : index`

    @parameter
    fn g(y: Int) -> Int:
        return x

    # expected-error @below {{cannot pass 'fn(y = index) capturing -> index' value, parameter expected 'fn(y = index) -> index'}}
    alias Bound = bind_fat_to_thin_target[g]
    Bound(3)
