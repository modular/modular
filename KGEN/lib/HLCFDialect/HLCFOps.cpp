//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//
static ParseResult
parseFor(OpAsmParser &p,
         SmallVectorImpl<OpAsmParser::UnresolvedOperand> &initArgs,
         SmallVectorImpl<Type> &initArgsTypes,
         SmallVectorImpl<Type> &resultTypes, Region &body) {
  SmallVector<OpAsmParser::Argument> loopArgs;

  // Parse the optional loop signature.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseOptionalRParen()) {
      OpAsmParser::Argument arg;
      OpAsmParser::UnresolvedOperand operand;
      auto parseEl = [&]() -> ParseResult {
        if (p.parseArgument(arg) || p.parseEqual() || p.parseOperand(operand) ||
            p.parseColonType(arg.type))
          return failure();
        loopArgs.push_back(arg);
        initArgs.push_back(operand);
        initArgsTypes.push_back(arg.type);
        return success();
      };
      if (p.parseCommaSeparatedList(parseEl) || p.parseRParen())
        return failure();
    }
    if (p.parseOptionalArrowTypeList(resultTypes))
      return failure();
  }
  return p.parseRegion(body, loopArgs);
}

static void printFor(OpAsmPrinter &p, Operation *op, ValueRange iterArgs,
                     TypeRange iterArgsTypes, TypeRange resultTypes,
                     Region &body) {
  if (!iterArgs.empty() || !resultTypes.empty()) {
    p << "(";
    llvm::interleaveComma(llvm::enumerate(iterArgs), p, [&](auto it) {
      auto [i, operand] = it;
      p << body.getArgument(i) << " = " << operand << " : " << iterArgsTypes[i];
    });
    p << ")";
    p.printOptionalArrowTypeList(resultTypes);
  }

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

void ForOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                  Value lowerBound, Value upperBound, Value step,
                  ValueRange iterArgs, std::optional<Attribute> unrollFactor,
                  M::HLCF::ForLoopBoundCmpPredicate cmpPredicateType,
                  M::HLCF::ForLoopIndVarCompute indVarComputeType) {
  MLIRContext *context = state.getContext();

  if (!unrollFactor)
    unrollFactor =
        HLCF::LoopUnrollFullAttr::get(context, HLCF::LoopUnrollFull::None);

  build(builder, state, results, lowerBound, upperBound, step, iterArgs,
        unrollFactor.value(),
        HLCF::ForLoopBoundCmpPredicateAttr::get(context, cmpPredicateType),
        HLCF::ForLoopIndVarComputeAttr::get(context, indVarComputeType));
}

LogicalResult ForOp::verify() {
  if (getIterArgs().size() != getBody().getNumArguments())
    return emitOpError("operand types do not match body region argument types");

  for (auto [loopArgs, blockarg] :
       llvm::zip(getIterArgs(), getBody().getArgumentTypes())) {
    if (loopArgs.getType() != blockarg)
      return emitOpError(
          "operand types do not match body region argument types");
  }

  for (auto [returnValueArg, resultType] :
       llvm::zip(getReturnValueArgs(), getResultTypes())) {
    if (returnValueArg.getType() != resultType)
      return emitOpError("operand types do not match return types");
  }

  return success();
}

void ForOp::getEntryTargets(ArrayRef<Attribute> operands,
                            SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());

  auto iter = dyn_cast_if_present<IntegerAttr>(operands.back());
  auto upperBound = this->getUpperBoundAsInt();
  auto step = this->getStepAsInt();

  if (!iter || !upperBound || !step) {
    targets.emplace_back(0, getOperands());
    targets.emplace_back(std::nullopt, getResults());
  }

  if ((step.value() > 0 && iter.getInt() < upperBound.value()) ||
      (step.value() < 0 && iter.getInt() > upperBound.value())) {
    // for-loop continues.
    targets.emplace_back(0, getOperands());
  } else {
    // for-loop exits.
    targets.emplace_back(std::nullopt, getResults());
  }
}

ValueRange ForOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0);
  return getBody().getArguments();
}

ErrorTreeOrSuccess ForOp::interpret(ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  state.transferControlFlowTo(&getBody().front(), operands);
  return success();
}

std::optional<int64_t> ForOp::getLowerBoundAsInt() {
  Value lowerBound = getLowerBound();
  IntegerAttr value;
  if (mlir::matchPattern(lowerBound, mlir::m_Constant(&value)))
    return value.getInt();
  return {};
}

std::optional<int64_t> ForOp::getUpperBoundAsInt() {
  Value upperBound = getUpperBound();
  IntegerAttr value;
  if (mlir::matchPattern(upperBound, mlir::m_Constant(&value)))
    return value.getInt();
  return {};
}

std::optional<int64_t> ForOp::getStepAsInt() {
  Value upperBound = getStep();
  IntegerAttr value;
  if (mlir::matchPattern(upperBound, mlir::m_Constant(&value)))
    return value.getInt();
  return {};
}

ValueRange ForOp::getReturnValueArgs() {
  return getIterArgs().drop_front().take_front(getNumResults());
}

// Get loop trip count.
std::optional<int64_t> ForOp::getTripCount() {
  std::optional<int64_t> lowerBound = getLowerBoundAsInt();
  std::optional<int64_t> upperBound = getUpperBoundAsInt();
  std::optional<int64_t> step = getStepAsInt();
  if (!lowerBound || !upperBound || !step)
    return {};

  return llvm::divideCeil(std::abs(upperBound.value() - lowerBound.value()),
                          std::abs(step.value()));
}

bool ForOp::isFullUnroll() {
  if (auto unroll = dyn_cast<HLCF::LoopUnrollFullAttr>(getUnrollFactorAttr()))
    return unroll.getValue() == HLCF::LoopUnrollFull::Full;
  return false;
}

std::optional<int64_t> ForOp::getUnrollFactorN() {
  if (auto n = dyn_cast<IntegerAttr>(getUnrollFactorAttr()))
    return n.getInt();

  return {};
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

/// arrow-type-list ::= `->` (`(` (type (`,` type)*)? `)`) | type
/// loop-arg ::= value `=` value `:` type
/// loop ::= (`(` (loop-arg (`,` loop-arg)*)? `)` arrow-type-list)? region
static ParseResult
parseLoop(OpAsmParser &p,
          SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
          SmallVectorImpl<Type> &operandTypes,
          SmallVectorImpl<Type> &resultTypes, Region &body) {
  SmallVector<OpAsmParser::Argument> loopArgs;

  // Parse the optional loop signature.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseOptionalRParen()) {
      OpAsmParser::Argument arg;
      OpAsmParser::UnresolvedOperand operand;
      auto parseEl = [&]() -> ParseResult {
        if (p.parseArgument(arg) || p.parseEqual() || p.parseOperand(operand) ||
            p.parseColonType(arg.type))
          return failure();
        loopArgs.push_back(arg);
        operands.push_back(operand);
        operandTypes.push_back(arg.type);
        return success();
      };
      if (p.parseCommaSeparatedList(parseEl) || p.parseRParen())
        return failure();
    }
    if (p.parseOptionalArrowTypeList(resultTypes))
      return failure();
  }
  return p.parseRegion(body, loopArgs);
}

static void printLoop(OpAsmPrinter &p, Operation *op, ValueRange operands,
                      TypeRange operandTypes, TypeRange resultTypes,
                      Region &body) {
  if (!operandTypes.empty() || !resultTypes.empty()) {
    p << " (";
    llvm::interleaveComma(llvm::enumerate(operands), p, [&](auto it) {
      auto [i, operand] = it;
      p << body.getArgument(i) << " = " << operand << " : " << operandTypes[i];
    });
    p << ")";
    p.printOptionalArrowTypeList(resultTypes);
  }
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult LoopOp::verify() {
  if (getOperandTypes() != getBody().getArgumentTypes())
    return emitOpError("operand types do not match body region argument types");
  return success();
}

void LoopOp::getEntryTargets(ArrayRef<Attribute> operands,
                             SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(0, getOperands());
}

ValueRange LoopOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0);
  return getBody().getArguments();
}

ErrorTreeOrSuccess LoopOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  state.transferControlFlowTo(&getBody().front(), operands);
  return success();
}

bool LoopOp::isFullUnroll() {
  if (!getUnrollFactor())
    return false;
  if (auto unroll = dyn_cast<HLCF::LoopUnrollFullAttr>(getUnrollFactorAttr()))
    return unroll.getValue() == HLCF::LoopUnrollFull::Full;
  return false;
}

std::optional<int64_t> LoopOp::getUnrollFactorN() {
  if (!getUnrollFactor())
    return {};
  if (auto n = dyn_cast<IntegerAttr>(getUnrollFactorAttr()))
    return n.getInt();
  return {};
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::getEntryTargets(ArrayRef<Attribute> operands,
                           SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  if (auto cond = dyn_cast_or_null<BoolAttr>(operands.front())) {
    targets.emplace_back(cond.getValue() ? 0 : 1);
  } else {
    targets.emplace_back(0);
    targets.emplace_back(1);
  }
}

ValueRange IfOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0 || *target == 1);
  return {};
}

ErrorTreeOrSuccess IfOp::interpret(ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  auto cond = dyn_cast_if_present<BoolAttr>(operands[0]);
  if (!cond)
    return ErrorTree(getLoc(), "non-constant condition");

  state.transferControlFlowTo(
      &(cond.getValue() ? getThenRegion() : getElseRegion()).front(), {});
  return success();
}

OpBuilder IfOp::getThenBodyBuilder() {
  assert(!getThenRegion().empty() && "Need a then block");
  return OpBuilder::atBlockEnd(&getThenRegion().front());
}

OpBuilder IfOp::getElseBodyBuilder() {
  assert(!getElseRegion().empty() && "Need an else block");
  return OpBuilder::atBlockEnd(&getElseRegion().front());
}

Block &IfOp::getThenBlock() { return getThenRegion().front(); }

Block &IfOp::getElseBlock() { return getElseRegion().front(); }

Operation *IfOp::getThenTerminator() { return getThenBlock().getTerminator(); }

Operation *IfOp::getElseTerminator() { return getElseBlock().getTerminator(); }

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

static ParseResult
parseSwitchCases(OpAsmParser &p, mlir::DenseI32ArrayAttr &caseValues,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int32_t> values;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    if (p.parseInteger(values.emplace_back()) ||
        p.parseRegion(*caseRegions.emplace_back(std::make_unique<Region>())))
      return failure();
  }
  caseValues = p.getBuilder().getDenseI32ArrayAttr(values);
  return success();
}

static void printSwitchCases(OpAsmPrinter &p, Operation *op,
                             ArrayRef<int32_t> caseValues,
                             MutableArrayRef<Region> caseRegions) {
  assert(caseValues.size() == caseRegions.size());
  for (auto [value, region] : llvm::zip(caseValues, caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(region);
  }
}

void SwitchOp::getEntryTargets(ArrayRef<Attribute> operands,
                               SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  if (auto cond = dyn_cast_or_null<IntegerAttr>(operands.front())) {
    for (auto [i, caseValue] : llvm::enumerate(getCaseValues())) {
      if (cond.getInt() == caseValue) {
        // Matching case branch.
        targets.emplace_back(i + 1);
        return;
      }
    }
    // Default branch.
    targets.emplace_back(0);
  } else {
    for (int32_t i = 0, e = getNumRegions(); i < e; ++i)
      targets.emplace_back(i);
  }
}

ValueRange SwitchOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  return {};
}

ErrorTreeOrSuccess SwitchOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  auto cond = dyn_cast_if_present<IntegerAttr>(operands[0]);
  if (!cond)
    return ErrorTree(getLoc(), "non-constant switch index");

  for (auto [i, caseValue] : llvm::enumerate(getCaseValues())) {
    if (cond.getInt() == caseValue) {
      // Matching case branch.
      state.transferControlFlowTo(&getCaseRegions()[i].front(), {});
      return success();
    }
  }
  // Default branch.
  state.transferControlFlowTo(&getDefaultRegion().front(), {});
  return success();
}

LogicalResult SwitchOp::verify() {
  if (!llvm::is_sorted(getCaseValues()))
    return emitOpError("expected case values to be sorted");
  DenseSet<int32_t> seenValues;
  for (int32_t caseValue : getCaseValues()) {
    if (!seenValues.insert(caseValue).second)
      return emitOpError("duplicate case value: ") << caseValue;
  }
  if (getCaseValues().size() != getCaseRegions().size()) {
    return emitOpError("has ") << getCaseValues().size() << " case values but "
                               << getCaseRegions().size() << " case regions";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

bool ContinueOp::isParentNode(Operation *op) {
  return isMatchingLoop(op, getLabelAttr());
}

void ContinueOp::getBranchTargets(ArrayRef<Attribute> operands,
                                  SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to the beginning of the body region.
  targets.emplace_back(0, getOperands());
}

ErrorTreeOrSuccess ContinueOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  LoopOp loop = getParentLoop(*this, getLabelAttr());
  state.transferControlFlowTo(&loop.getBody().front(), operands);
  return success();
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

void BreakOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (!isMatchingLoop((*this)->getParentOp(), getLabelAttr()))
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

mlir::Speculation::Speculatability BreakOp::getSpeculatability() {
  return isMatchingLoop((*this)->getParentOp(), getLabelAttr())
             ? mlir::Speculation::Speculatable
             : mlir::Speculation::NotSpeculatable;
}

bool BreakOp::isParentNode(Operation *op) {
  return isMatchingLoop(op, getLabelAttr());
}

void BreakOp::getBranchTargets(ArrayRef<Attribute> operands,
                               SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the loop operation.
  targets.emplace_back(std::nullopt, getOperands());
}

ErrorTreeOrSuccess BreakOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  LoopOp loop = getParentLoop(*this, getLabelAttr());
  state.setReturnValues(operands);
  state.transferControlFlowTo(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

bool YieldOp::isParentNode(Operation *op) { return isa<IfOp, SwitchOp>(op); }

void YieldOp::getBranchTargets(ArrayRef<Attribute> operands,
                               SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the parent operation.
  targets.emplace_back(std::nullopt, getOperands());
}

ErrorTreeOrSuccess YieldOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  state.setReturnValues(operands);
  state.transferControlFlowTo((*this)->getParentOp());
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

bool ForYieldOp::isParentNode(Operation *op) { return isa<ForOp>(op); }

void ForYieldOp::getBranchTargets(ArrayRef<Attribute> operands,
                                  SmallVectorImpl<ControlFlowTarget> &targets) {

  assert(operands.size() == getNumOperands());

  ForOp forLoop = this->getParentOp<ForOp>();
  // Branch to the beginning of the body region.
  targets.emplace_back(0, getOperands());
  // Though `hlcf.for.yield` can exit when iter count meets upperbound.
  targets.emplace_back(std::nullopt, getOperands().drop_front().take_front(
                                         forLoop.getNumResults()));
}

ErrorTreeOrSuccess ForYieldOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  ForOp forLoop = this->getParentOp<ForOp>();
  auto iter = dyn_cast_or_null<IntegerAttr>(
      state.lookupValue(getForInductionVariableOperand()));
  if (!iter)
    return ErrorTree(getLoc(), "non-integer induction variable.");

  auto upperBound =
      dyn_cast_or_null<IntegerAttr>(state.lookupValue(forLoop.getUpperBound()));
  if (!upperBound)
    return ErrorTree(getLoc(), "non-integer parent for-loop upperBound.");

  auto step =
      dyn_cast_or_null<IntegerAttr>(state.lookupValue(forLoop.getStep()));
  if (!step)
    return ErrorTree(getLoc(), "non-integer parent for-loop step.");

  bool continueFor =
      (step.getInt() > 0 && iter.getInt() < upperBound.getInt()) ||
      (step.getInt() < 0 && iter.getInt() > upperBound.getInt());

  if (continueFor)
    state.transferControlFlowTo(&forLoop.getBody().front(), operands);
  else
    state.transferControlFlowTo(forLoop->getParentOp());
  return success();
}

Value ForYieldOp::getForInductionVariableOperand() {
  return this->getOperands().front();
}

LogicalResult ForYieldOp::verify() {
  ForOp parentFor = getParentOp<ForOp>();

  if (getOperands().size() != parentFor.getBody().getNumArguments())
    return emitOpError("operand types do not match parent for-loop's body "
                       "region argument types");

  if (getReturnValues().size() != parentFor.getNumResults())
    return emitOpError("number of operands in return value segment do not "
                       "match parent for-loop's number of results.");

  for (auto [parentOperand, operand] :
       llvm::zip(parentFor.getIterArgs(), getOperands())) {
    if (parentOperand.getType() != operand.getType())
      return emitOpError(
          "operand types do not match parent for-loop's operand types");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/HLCFDialect/HLCF.cpp.inc"
