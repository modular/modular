//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file implements the GraphRT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace M;
using namespace GRT;

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

void ChainCreateOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState) {
  build(odsBuilder, odsState, odsBuilder.getType<MEF::ChainType>(), {});
}

ParseResult ChainCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  auto chainType = MEF::ChainType::get(parser.getContext());

  if (parseParenOperandListWithDefaultType(parser, result, chainType) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Result type is deterministic
  result.types.push_back(chainType);

  return success();
}

void ChainCreateOp::print(OpAsmPrinter &p) {
  auto chainType = MEF::ChainType::get(getContext());
  p << ' ';
  printParenOperandListWithDefaultType(p, getOperands(), chainType);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
}

static LogicalResult eliminateImpliedByChains(ChainCreateOp createOp,
                                              PatternRewriter &rewriter) {
  auto isGatingOperand = [](const OpOperand *opOperand) {
    Operation *op = opOperand->getOwner();

    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Only the extra strict operands are gating for control flow ops.
      auto extraOperands = cfOp.getStrictExtraOperands();
      return llvm::is_contained(extraOperands, opOperand->get());
    }

    return !isa<MEF::NonStrictInterface>(op);
  };

  llvm::SmallDenseSet<Value> impliedByOperand;

  for (auto chain : createOp.getInputChains()) {
    auto producer = chain.getDefiningOp();
    if (!producer)
      continue;

    for (OpOperand &operand : producer->getOpOperands()) {
      if (isGatingOperand(&operand))
        impliedByOperand.insert(operand.get());
    }
  }

  SmallVector<Value> minimizedOperands;
  minimizedOperands.reserve(createOp.getInputChains().size());

  for (auto chain : createOp.getInputChains()) {
    if (!impliedByOperand.contains(chain))
      minimizedOperands.push_back(chain);
  }

  if (minimizedOperands.size() == createOp.getNumOperands())
    return failure();

  rewriter.modifyOpInPlace(createOp,
                           [&] { createOp.setInputChains(minimizedOperands); });

  return success();
}

LogicalResult ChainCreateOp::canonicalize(ChainCreateOp createOp,
                                          PatternRewriter &rewriter) {
  llvm::SetVector<Value> operands;
  for (auto operand : createOp.getOperands()) {
    // Ignore empty chain.create operands
    if (auto innerChainCreate = operand.getDefiningOp<ChainCreateOp>()) {
      if (innerChainCreate.getNumOperands() == 0)
        continue;
    }
    operands.insert(operand);
  }
  // Replace chain.create(x) with x provided the types agree.
  if (operands.size() == 1 &&
      operands.front().getType() == createOp.getResult().getType()) {
    rewriter.replaceOp(createOp, operands.front());
    return success();
  }
  // Remove duplicate operands.
  if (operands.size() < createOp.getOperands().size()) {
    rewriter.modifyOpInPlace(createOp, [&] {
      createOp.getOperandsMutable().assign(operands.takeVector());
    });
    return success();
  }

  return eliminateImpliedByChains(createOp, rewriter);
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the target attribute was specified.
  MEF::CallableOp fn = symbolTable.lookupNearestSymbolFrom<MEF::CallableOp>(
      *this, getTargetAttr());
  if (!fn)
    return emitOpError() << "'" << getTargetAttr()
                         << "' does not reference a valid invokable";

  // Verify that the operand and result types match the target.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for target");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for target");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  OpAsmParser::UnresolvedOperand condOperand;
  Region *thenRegion = state.addRegion();
  Region *elseRegion = state.addRegion();
  if (parser.parseOperand(condOperand) ||
      parser.resolveOperand(condOperand,
                            IntegerType::get(state.getContext(), 1),
                            state.operands) ||
      parseParenOperandListWithShadowing(parser, state, argumentInfo,
                                         /*isIsolatedFromAbove=*/true) ||
      parser.parseArrowTypeList(state.types) ||
      parser.parseOptionalAttrDictWithKeyword(state.attributes) ||
      parseRegion(parser, argumentInfo, *thenRegion,
                  /*isIsolatedFromAbove=*/true) ||
      parser.parseKeyword("else") ||
      parseRegion(parser, argumentInfo, *elseRegion,
                  /*isIsolatedFromAbove=*/true))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << ' ' << getCond() << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getThenRegion().getArguments(),
                                     /*isIsolatedFromAbove=*/true);
  p.printArrowTypeList(getResultTypes());
  p << ' ';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  printRegion(p, getInputs(), getThenRegion(), /*isIsolatedFromAbove=*/true);
  p << " else ";
  printRegion(p, getInputs(), getElseRegion(), /*isIsolatedFromAbove=*/true);
}

void IfOp::build(
    OpBuilder &builder, OperationState &result, Value cond, ValueRange inputs,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)> thenBuilder,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)> elseBuilder) {
  result.addOperands(cond);
  result.addOperands(inputs);

  SmallVector<Location> argLocations(inputs.size(), result.location);

  // Build then region.
  OpBuilder::InsertionGuard guard(builder);

  Region *thenRegion = result.addRegion();
  Block *thenBlock = builder.createBlock(thenRegion);
  thenBlock->addArguments(TypeRange(inputs), argLocations);
  thenBuilder(builder, result.location, thenBlock->getArguments());

  // Add a terminator if not present
  if (!thenBlock->mightHaveTerminator())
    IfOp::ensureTerminator(*thenRegion, builder, result.location);

  // Build else region.
  Region *elseRegion = result.addRegion();
  auto elseBlock = builder.createBlock(elseRegion);
  elseBlock->addArguments(TypeRange(inputs), argLocations);
  elseBuilder(builder, result.location, elseBlock->getArguments());

  // Add a terminator if not present
  if (!elseBlock->mightHaveTerminator())
    IfOp::ensureTerminator(*elseRegion, builder, result.location);

  // Pull the result types off of the terminator.
  auto thenResults = thenBlock->getTerminator()->getOperandTypes();
  result.addTypes(thenResults);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  Region *condRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();
  if (parseParenOperandListWithShadowing(parser, result, argumentInfo,
                                         /*isIsolatedFromAbove=*/true) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegion(parser, argumentInfo, *condRegion,
                  /*isIsolatedFromAbove=*/true) ||
      parser.parseKeyword("do") ||
      parseRegion(parser, argumentInfo, *bodyRegion,
                  /*isIsolatedFromAbove=*/true))
    return failure();

  // The op's return types match those of the signature arguments.
  result.types.reserve(result.operands.size());
  for (auto operand : result.operands)
    result.types.push_back(operand.getType());

  return success();
}

void WhileOp::print(OpAsmPrinter &p) {
  p << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getCondRegion().getArguments(),
                                     /*isIsolatedFromAbove=*/true);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegion(p, getInputs(), getCondRegion(), /*isIsolatedFromAbove=*/true);
  p << " do ";
  printRegion(p, getInputs(), getBodyRegion(), /*isIsolatedFromAbove=*/true);
}

LogicalResult WhileOp::verify() {
  // The 'cond' block should return an additional i1.
  Block *condBlock = &getCondRegion().front();
  auto condOutput = cast<MEF::OutputOp>(condBlock->getTerminator());
  if (condOutput->getNumOperands() != getInputs().size() + 1 ||
      condOutput->getOperand(getInputs().size()).getType() !=
          IntegerType::get(getContext(), 1))
    return emitOpError() << "expected 'cond' block to return additional i1 "
                            "result for loop condition";

  // The 'body' block should not return any additional results.
  Block *bodyBlock = &getBodyRegion().front();
  if (bodyBlock->getNumArguments() != getInputs().size())
    return emitOpError()
           << "expected 'body' block to return no additional results";

  return success();
}

MEF::OutputOp WhileOp::getCondTerminator() {
  return cast<MEF::OutputOp>(getCondRegion().front().getTerminator());
}

MEF::OutputOp WhileOp::getBodyTerminator() {
  return cast<MEF::OutputOp>(getBodyRegion().front().getTerminator());
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

ParseResult RepeatI32Op::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  OpAsmParser::UnresolvedOperand countOperand;
  Region *body = result.addRegion();
  if (parser.parseOperand(countOperand) ||
      parser.resolveOperand(
          countOperand,
          IntegerType::get(result.getContext(), 32, IntegerType::Signed),
          result.operands) ||
      parseParenOperandListWithShadowing(parser, result, argumentInfo,
                                         /*isIsolatedFromAbove=*/true) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegion(parser, argumentInfo, *body, /*isIsolatedFromAbove=*/true))
    return failure();

  // The result types are the same as the operands (ignoring 'tripCount').
  result.types.reserve(result.operands.size() - 1);
  for (auto operand : llvm::drop_begin(result.operands))
    result.types.push_back(operand.getType());

  return success();
}

void RepeatI32Op::print(OpAsmPrinter &p) {
  p << ' ' << getTripCount() << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getBodyRegion().getArguments(),
                                     /*isIsolatedFromAbove=*/true);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegion(p, getInputs(), getBodyRegion(), /*isIsolatedFromAbove=*/true);
}

LogicalResult RepeatI32Op::verify() {
  // The 'body' block should not return any additional results.
  Block *bodyBlock = &getBodyRegion().front();
  if (bodyBlock->getNumArguments() != getInputs().size())
    return emitOpError()
           << "expected 'body' block to return no additional results";

  return success();
}

//===----------------------------------------------------------------------===//
// TestcaseOp
//===----------------------------------------------------------------------===//

ParseResult TestcaseOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse just the symbol name first.
  StringAttr symbol;
  if (parser.parseSymbolName(symbol, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  // Parse the body region. It has no arguments whatsoever.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, false))
    return failure();

  return success();
}

void TestcaseOp::print(OpAsmPrinter &p) { p.printRegion(getBody()); }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "MLRT/GraphRT/GRTDialect/GRT.cpp.inc"
