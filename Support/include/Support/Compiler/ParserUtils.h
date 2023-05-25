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

/// Parses a parenthesized operand list with optional 'as' clauses. Each
/// operand contributes both to the operation state's operand list and to
/// the argumentInfo array to be used by the op's nested blocks.
///
/// Eg: Basic form:
///   (%a: i32, %b: f32)
/// Eg: With corresponding block argument name distinct from the operand name
///   (%a: i32, %a as %b: f32)
///
/// Intended to pair with parseRegionWithShadowing for nested regions.
ParseResult parseParenOperandListWithShadowing(
    OpAsmParser &parser, OperationState &state,
    SmallVectorImpl<OpAsmParser::Argument> &argumentInfo);

/// Parses a parenthesized operand list with optional type annotations.
/// The given optional type is used if no type annotation is given.
///
/// Eg: Basic form:
///   (%a: i32, %b: f32)
/// Eg: With default type taken from defaultType:
///   (%a, %b: f32)
ParseResult parseParenOperandListWithDefaultType(OpAsmParser &parser,
                                                 OperationState &state,
                                                 Type defaultType);

/// Parses a region, using argumentInfo as the block arguments, which are
/// allowed to shadow outer SSA values.
///
/// The operands and argumentInfo are expected to have come from
/// parseParenOperandList.
ParseResult
parseRegionWithShadowing(OpAsmParser &parser,
                         const SmallVector<OpAsmParser::Argument> &argumentInfo,
                         Region &region);

/// Prints a parenthesized operand list, matching the syntax parsed by
/// parseParenOperandListWithShadowing.
///
/// Intended to pair with printRegionWithShadowing for nested regions.
void printParenOperandListWithShadowing(
    OpAsmPrinter &printer, const OperandRange &operands,
    const Block::BlockArgListType &arguments);

/// Prints a parenthesized operand list, matching the syntax parsed by
/// parseParenOperandListWithDefaultType.
void printParenOperandListWithDefaultType(OpAsmPrinter &printer,
                                          const OperandRange &operands,
                                          Type defaultType);

/// Prints a region. If possible, block argument names will shadow the names
/// of operands.
void printRegionWithShadowing(OpAsmPrinter &printer,
                              const OperandRange &operands, Region &region);

} // namespace M

#endif // SUPPORT_COMPILER_COMPILERUTILS_H
