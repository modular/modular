//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_COMPILERUTILS_H
#define SUPPORT_COMPILER_COMPILERUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

namespace M {

/// Parse a parenthesized operand list with types included, e.g.
/// `(%a: i32, %b: f32)`. For operands without type annotation, use
/// `optDefaultType`.
ParseResult
parseParenOperandList(OpAsmParser &parser, OperationState &result,
                      SmallVectorImpl<OpAsmParser::Argument> &argumentInfo,
                      Type optDefaultType = {});

} // namespace M

#endif // SUPPORT_COMPILER_COMPILERUTILS_H
