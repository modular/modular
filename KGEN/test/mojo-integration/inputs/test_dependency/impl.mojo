# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias int = __mlir_type.index
alias `0` = __mlir_attr.`0 : index`

alias ALIAS = child()


fn use_me() -> int:
    @no_inline
    @parameter
    fn preserve_me() -> int:
        return ALIAS

    return preserve_me()


fn child() -> int:
    return `0`


fn dead() -> int:
    return `0`
