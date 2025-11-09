# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias ALIAS = child()


fn use_me() -> __mlir_type.index:
    @no_inline
    @parameter
    fn preserve_me() -> __mlir_type.index:
        return ALIAS

    return preserve_me()


fn child() -> __mlir_type.index:
    return __mlir_attr.`0 : index`


fn dead() -> __mlir_type.index:
    return __mlir_attr.`0 : index`
