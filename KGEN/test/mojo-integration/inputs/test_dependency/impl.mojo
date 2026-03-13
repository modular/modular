# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

comptime ALIAS = child()


def use_me() -> __mlir_type.index:
    @no_inline
    @parameter
    def preserve_me() -> __mlir_type.index:
        return ALIAS

    return preserve_me()


def child() -> __mlir_type.index:
    return __mlir_attr.`0 : index`


def dead() -> __mlir_type.index:
    return __mlir_attr.`0 : index`
