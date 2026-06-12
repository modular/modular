# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s --debug-level full 2>&1 | FileCheck %s


@no_inline
def _trait_is_eq[t1: type_of(AnyType), t2: type_of(AnyType)]() -> Bool:
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        `#kgen.type<`,
        +t1,
        `> : !kgen.type`,
        `,`,
        `#kgen.type<`,
        +t2,
        `> : !kgen.type`,
        `> : !kgen.scalar<bool>`,
    ]


def main():
    # CHECK: True
    print(_trait_is_eq[AnyType, AnyType]())
    # CHECK-NEXT: False
    print(_trait_is_eq[Copyable, AnyType]())
