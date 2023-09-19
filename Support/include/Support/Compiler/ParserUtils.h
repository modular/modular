//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_COMPILERUTILS_H
#define SUPPORT_COMPILER_COMPILERUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringExtras.h"

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

/// Parses a 'buffer signature' of the form:
///    ( in|out|mut %x : type, ... )
/// The SSA values will be added to buffers, their types to bufferTypes,
/// and the inOutSignatureAttr will have matching arity and capture the
/// in/out/mut keywords.
///
/// Note that the types are unconstrained and need not be any particular
/// 'buffer' type. However generally they are pointer-like for the in/out/mut
/// keyword to be necessary.
ParseResult
parseBufferSignature(OpAsmParser &parser,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand> &buffers,
                     SmallVectorImpl<Type> &bufferTypes,
                     InOutSignatureAttr &inOutSignatureAttr);

/// Prints a 'buffer signature', matching the syntax parsed by
/// parseBufferSignature.
void printBufferSignature(OpAsmPrinter &printer, const Operation *opIgnored,
                          ValueRange buffers, TypeRange bufferTypes,
                          InOutSignatureAttr inOutSignatureAttr);

/// This is an AsmPrinter implementation that just outputs to an external output
/// stream.
class StreamAsmPrinter : public AsmPrinter {
public:
  explicit StreamAsmPrinter(raw_ostream &os) : os(os) {}

  /// Implement all the virtual hooks.

  raw_ostream &getStream() const override { return os; }

  /// Trivial hooks

  void printType(Type type) override { os << type; }
  void printAttribute(Attribute attr) override { os << attr; }
  void printAttributeWithoutType(Attribute attr) override {
    attr.print(os, /*elideType=*/true);
  }
  LogicalResult printAlias(Attribute attr) override { return failure(); }
  LogicalResult printAlias(Type type) override { return failure(); }

  /// Less trivial hooks.

  void printString(StringRef string) override;
  void printKeywordOrString(StringRef keyword) override;
  void printSymbolName(StringRef symbolRef) override;
  void
  printResourceHandle(const mlir::AsmDialectResourceHandle &resource) override;

  /// Print floats like MLIR does.
  void printFloat(const APFloat &value) override;

private:
  /// The stream to output to.
  raw_ostream &os;
};

} // namespace M

#endif // SUPPORT_COMPILER_COMPILERUTILS_H
