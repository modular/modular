# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -elaborate -O0 %s -S 2>&1 | FileCheck %s

# A deferred f-string op whose template is only well-formed once parameters
# bind: the parser cannot reject it, so the elaborator's `lowerFStringMLIROp`
# re-parse is what surfaces the malformed assembly after binding.


# CHECK: failed to parse f-string MLIR op
def bogus[
    T: __mlir_type.`!kgen.dtype`
](x: __mlir_type[`!kgen.scalar<`, T, `>`]) -> __mlir_type[
    `!kgen.scalar<`, T, `>`
]:
    return __mlir_op[`pop.add totally bogus %{x} : %{type_of(x)}`]


@export
def top(
    a: __mlir_type.`!kgen.scalar<si32>`,
) abi("Mojo") -> __mlir_type.`!kgen.scalar<si32>`:
    return bogus[__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`](a)
