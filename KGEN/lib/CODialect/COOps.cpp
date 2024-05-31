//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/CODialect/CODialect.h"
#include "KGEN/CODialect/COUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"

using namespace M;
using namespace KGEN;
using namespace CO;

//===----------------------------------------------------------------------===//
// HandleOp
//===----------------------------------------------------------------------===//

LogicalResult HandleOp::verify() {
  if (auto func = (*this)->getParentOfType<FuncOp>()) {
    if (func.getNumResults() != 1) {
      return emitOpError("surrounding function must have 1 result")
                 .attachNote(func.getLoc())
             << "see function here";
    }
    Type resultType = func.getResultTypes().front();
    if (resultType != getType()) {
      return emitOpError("surrounding function result type does not match "
                         "coroutine handle type")
                 .attachNote(func.getLoc())
             << "surrounding function returns " << resultType;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SuspendOp
//===----------------------------------------------------------------------===//

static ParseResult parseSuspendBody(OpAsmParser &p, Region &body) {
  OpAsmParser::Argument arg;
  if (p.parseArgument(arg))
    return failure();
  arg.type = CoroutineType::get(p.getContext());
  return p.parseRegion(body, arg);
}

static void printSuspendBody(OpAsmPrinter &p, Operation *op, Region &body) {
  p.printRegionArgument(body.getArgument(0), /*argAttrs=*/{},
                        /*omitType=*/true);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

void SuspendOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // Sugar the SSA value name
  setNameFn(region.getArgument(0), "hdl");
}

LogicalResult SuspendOp::verify() {
  Region &body = getBody();
  if (body.getNumArguments() == 1 &&
      isa<CoroutineType>(body.getArgument(0).getType()))
    return success();
  return emitOpError("expected its body region to have a "
                     "single `!co.routine` type argument");
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

LogicalResult InvokeOp::verify() {
  if (!cast<SignatureType>(getCallee().getType()).isAsync())
    return emitOpError("callable must be 'async'");
  return success();
}

static ParseResult parseAsyncParametricCallee(
    OpAsmParser &p, TypedAttr &callee,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes) {
  if (failed(parseParametricCallee(p, callee)))
    return failure();
  SignatureType calleeSignature = cast<SignatureType>(callee.getType());
  // Operands match signature arguments with the exception that byref error and
  // byref result are omitted.
  unsigned start = 0;
  if (calleeSignature.isThrows())
    ++start;
  if (calleeSignature.hasInitSelfArg() || calleeSignature.hasMemoryOnlyResult())
    ++start;
  unsigned i = 0;
  ArrayRef<Type> argumentTypes(calleeSignature.getArguments().slice(start));
  if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                       [&]() -> ParseResult {
                                         OpAsmParser::UnresolvedOperand operand;
                                         if (failed(p.parseOperand(operand)))
                                           return failure();
                                         operands.push_back(operand);
                                         Type type = argumentTypes[i];
                                         ++i;
                                         operandTypes.push_back(type);
                                         return success();
                                       })))
    return failure();
  return success();
}

static void printAsyncParametricCallee(OpAsmPrinter &p, Operation *op,
                                       TypedAttr callee, ValueRange operands,
                                       TypeRange operandTypes) {
  printParametricCallee(p, op, callee);
  p << "(";
  p.printOperands(operands);
  p << ")";
}

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

ArrayRef<Type> ExecuteOp::getResultTypes() { return getTypes(); }

//===----------------------------------------------------------------------===//
// CODialect
//===----------------------------------------------------------------------===//

void CODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/CODialect/CO.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/CODialect/CO.cpp.inc"
