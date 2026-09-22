//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "mlir/IR/Matchers.h"

using namespace M;
using namespace M::KGEN;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// parseLoop / printLoop
//===----------------------------------------------------------------------===//

/// arrow-type-list ::= `->` (`(` (type (`,` type)*)? `)`) | type
/// loop-arg ::= value `=` value `:` type
/// loop ::= (`(` (loop-arg (`,` loop-arg)*)? `)` arrow-type-list)? region
ParseResult
HLCF::parseLoop(OpAsmParser &p,
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

void HLCF::printLoop(OpAsmPrinter &p, Operation *op, ValueRange operands,
                     TypeRange operandTypes, TypeRange resultTypes,
                     Region &body) {
  if (!operandTypes.empty() || !resultTypes.empty()) {
    p << " (";
    llvm::interleaveComma(llvm::enumerate(operands), p, [&](auto it) {
      auto [i, operand] = it;
      p.printRegionArgument(body.getArgument(i), /*argAttrs=*/{},
                            /*omitType=*/true);
      p << " = " << operand << " : " << operandTypes[i];
    });
    p << ")";
    p.printOptionalArrowTypeList(resultTypes);
  }
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ForOp bounds custom assembly
//===----------------------------------------------------------------------===//

ParseResult HLCF::parseForBoundsWithOptionalType(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &lowerBound,
    Type &lowerBoundType, OpAsmParser::UnresolvedOperand &upperBound,
    Type &upperBoundType, OpAsmParser::UnresolvedOperand &step,
    Type &stepType) {
  if (parser.parseOperand(lowerBound) || parser.parseKeyword("to") ||
      parser.parseOperand(upperBound) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();
  // Optional `: <type>` annotation — default to `index` for backward compat.
  Type boundsType = parser.getBuilder().getIndexType();
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(boundsType))
      return failure();
  }
  stepType = boundsType;
  lowerBoundType = boundsType;
  upperBoundType = boundsType;
  return success();
}

void HLCF::printForBoundsWithOptionalType(OpAsmPrinter &printer, Operation *,
                                          Value lowerBound, Type lowerBoundType,
                                          Value upperBound, Type /*ubType*/,
                                          Value step, Type /*stepType*/) {
  printer << lowerBound << " to " << upperBound << " step " << step;
  // Omit type annotation for `index` to preserve the compact legacy format.
  if (!lowerBoundType.isIndex())
    printer << " : " << lowerBoundType;
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

LogicalResult ForOp::verify() {
  // Lower bound, upper bound, and step must all have the same type.
  Type boundsType = getLowerBound().getType();
  if (getUpperBound().getType() != boundsType) {
    return emitOpError("upper bound type '")
           << getUpperBound().getType() << "' does not match lower bound type '"
           << boundsType << "'";
  }
  if (getStep().getType() != boundsType) {
    return emitOpError("step type '")
           << getStep().getType() << "' does not match lower bound type '"
           << boundsType << "'";
  }

  if (getIterArgs().size() != getBody().getNumArguments())
    return emitOpError("operand count do not match body region argument count");

  for (auto [i, loopArg, blockArg] :
       llvm::enumerate(getIterArgs(), getBody().getArguments())) {
    if (loopArg.getType() != blockArg.getType()) {
      return emitOpError("operand #")
             << i << " type " << loopArg.getType()
             << " does not match type of corresponding block argument "
             << blockArg.getType();
    }
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
  std::optional<int64_t> upperBound = getUpperBoundAsInt();
  std::optional<int64_t> step = getStepAsInt();

  if (!iter || !upperBound || !step) {
    targets.emplace_back(0, getIterArgs());
    targets.emplace_back(std::nullopt, getResults());
    return;
  }

  if ((step.value() > 0 && iter.getInt() < upperBound.value()) ||
      (step.value() < 0 && iter.getInt() > upperBound.value())) {
    // for-loop continues.
    targets.emplace_back(0, getIterArgs());
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
  return state.transferControlFlowTo(getBody(), operands);
}

ErrorTreeOrSuccess
ForOp::parametric_interpret(ArrayRef<Attribute> operands,
                            ParametricInterpreterState &state) {
  return interpret(operands, state);
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

  int64_t r = upperBound.value() - lowerBound.value();

  ForLoopBoundCmpPredicate pred = getCmpPredicateType();
  ForLoopIndVarCompute opType = getIndVarComputeType();
  // When lowerBound and upperBound don't form a valid range, return 0.
  switch (opType) {
  case ForLoopIndVarCompute::ADD:
    if (step.value() > 0 && r < 0)
      return 0;
    if (step.value() < 0 && r > 0)
      return 0;
    break;
  case ForLoopIndVarCompute::SUB:
    if (step.value() > 0 && r > 0)
      return 0;
    if (step.value() < 0 && r < 0)
      return 0;
  }

  r = std::abs(r);
  if (pred == ForLoopBoundCmpPredicate::SGE ||
      pred == ForLoopBoundCmpPredicate::SLE)
    r += 1;

  return llvm::divideCeil(r, std::abs(step.value()));
}

bool ForOp::isFullUnroll() { return getUnrollLevel().isFull(); }

std::optional<int64_t> ForOp::getUnrollFactorN() {
  UnrollLevel level = getUnrollLevel();
  if (level.isFactor())
    return level.getFactor();
  return {};
}

void ForOp::insertVariants(ValueRange newOperands) {
  // Add the variant values to both the result argument and the body iter
  // argument ranges.
  MutableOperandRange resultArgs =
      getIterArgsMutable().slice(1, getNumResults());
  resultArgs.append(newOperands);

  size_t leading = 1 + resultArgs.size();
  MutableOperandRange iterArgs =
      getIterArgsMutable().slice(leading, getIterArgs().size() - leading);
  iterArgs.append(newOperands);
}

BlockArgument ForOp::insertArgumentToRegion(Location loc, Type argType,
                                            size_t argIdx, Region &region) {
  // Add argument to match both retValues and otherIterValues segments of the
  // ForOp with variants added as new operands.
  region.insertArgument(1 + getNumResults() + argIdx, argType, loc);
  return region.addArgument(argType, loc);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

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
  return state.transferControlFlowTo(getBody(), operands);
}

ErrorTreeOrSuccess
LoopOp::parametric_interpret(ArrayRef<Attribute> operands,
                             ParametricInterpreterState &state) {
  return interpret(operands, state);
}

bool LoopOp::isFullUnroll() {
  HLCF::UnrollLevelAttr level =
      dyn_cast_if_present<HLCF::UnrollLevelAttr>(getUnrollLevelAttr());
  if (!level)
    return false;
  return level.getValue().isFull();
}

std::optional<int64_t> LoopOp::getUnrollFactorN() {
  HLCF::UnrollLevel level = getUnrollLevelValue();
  if (level.isNone())
    return {};
  return level.getFactor();
}

HLCF::UnrollLevel LoopOp::getUnrollLevelValue() {
  if (auto unrollAttr =
          dyn_cast_if_present<HLCF::UnrollLevelAttr>(getUnrollLevelAttr()))
    return unrollAttr.getValue();
  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(getUnrollLevelAttr()))
    return (int32_t)intAttr.getInt();
  return 0;
}

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
      return state.transferControlFlowTo(getCaseRegions()[i], {});
    }
  }
  // Default branch.
  return state.transferControlFlowTo(getDefaultRegion(), {});
}

ErrorTreeOrSuccess
SwitchOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  return interpret(operands, state);
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
  return state.transferControlFlowTo(loop.getBody(), operands);
}

ErrorTreeOrSuccess
ContinueOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {
  return interpret(operands, state);
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
  return state.transferControlFlowTo(loop, operands);
}

ErrorTreeOrSuccess
BreakOp::parametric_interpret(ArrayRef<Attribute> operands,
                              ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

bool YieldOp::isParentNode(Operation *op) {
  if (isa<SwitchOp, IfOp>(op))
    return true;
  // Yield in a match targets only the else region; case regions use
  // match.next / match.complete instead.
  if (auto match = dyn_cast<MatchOp>(op))
    return !match.containsInCaseRegion(*this);
  if (auto match = dyn_cast<ComptimeMatchOp>(op))
    return !match.containsInCaseRegion(*this);
  return false;
}

void YieldOp::getBranchTargets(ArrayRef<Attribute> operands,
                               SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the parent operation.
  targets.emplace_back(std::nullopt, getOperands());
}

ErrorTreeOrSuccess YieldOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  return state.transferControlFlowTo((*this)->getParentOp(), operands);
}

ErrorTreeOrSuccess
YieldOp::parametric_interpret(ArrayRef<Attribute> operands,
                              ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// ForYieldOp
//===----------------------------------------------------------------------===//

bool ForYieldOp::isParentNode(Operation *op) { return isa<ForOp>(op); }

void ForYieldOp::getBranchTargets(ArrayRef<Attribute> operands,
                                  SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  ForOp forLoop = getParentOp<ForOp>();
  auto iter = dyn_cast_or_null<IntegerAttr>(operands.front());

  std::optional<int64_t> upperBound = forLoop.getUpperBoundAsInt();
  std::optional<int64_t> step = forLoop.getStepAsInt();
  if (!iter || !upperBound || !step) {
    // Branch to the beginning of the body region.
    targets.emplace_back(0, getOperands());
    // Though `hlcf.for.yield` can exit when iter count meets upperbound.
    targets.emplace_back(std::nullopt, getReturnValues());
    return;
  }

  ForLoopBoundCmpPredicate pred = forLoop.getCmpPredicateType();
  ForLoopIndVarCompute opType = forLoop.getIndVarComputeType();

  bool continueFor = (step.value() > 0 && iter.getInt() < upperBound.value() &&
                      opType == ForLoopIndVarCompute::ADD) ||
                     (step.value() < 0 && iter.getInt() > upperBound.value() &&
                      opType == ForLoopIndVarCompute::ADD) ||
                     (step.value() < 0 && iter.getInt() < upperBound.value() &&
                      opType == ForLoopIndVarCompute::SUB) ||
                     (step.value() > 0 && iter.getInt() > upperBound.value() &&
                      opType == ForLoopIndVarCompute::SUB);

  continueFor |= (iter.getInt() == upperBound.value() &&
                  pred == ForLoopBoundCmpPredicate::SGE) ||
                 (iter.getInt() == upperBound.value() &&
                  pred == ForLoopBoundCmpPredicate::SLE);

  if (continueFor) {
    // Branch to the beginning of the body region if continues.
    targets.emplace_back(0, getOperands());
  } else {
    // Though `hlcf.for.yield` can exit when iter count meets upperbound.
    targets.emplace_back(std::nullopt, getReturnValues());
  }
}

void ForYieldOp::insertVariants(ValueRange newOperands) {
  // Append variants to both retValues and otherIterValues.
  getReturnValuesMutable().append(newOperands);
  getOtherIterValuesMutable().append(newOperands);
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
// IfOp
//===----------------------------------------------------------------------===//

static ParseResult
parseIf(OpAsmParser &parser, Region &thenRegion,
        SmallVectorImpl<std::unique_ptr<Region>> &elifRegionsRegions,
        Region &elseRegion) {
  // First then region (condition is an SSA operand parsed by ODS).
  if (failed(parser.parseRegion(thenRegion)))
    return failure();

  // Each subsequent arm is introduced by `else`. If `then` follows the
  // region, it is an additional (cond, then) pair; otherwise it is the final
  // else region.
  while (true) {
    if (failed(parser.parseKeyword("else")))
      return failure();

    SmallVector<OpAsmParser::Argument> regionArgs;
    if (failed(parser.parseArgumentList(
            regionArgs, AsmParser::Delimiter::OptionalParen, true, false)))
      return failure();

    auto region = std::make_unique<Region>();
    if (failed(parser.parseRegion(*region, regionArgs)))
      return failure();

    if (succeeded(parser.parseOptionalKeyword("then"))) {
      elifRegionsRegions.push_back(std::move(region));

      SmallVector<OpAsmParser::Argument> thenArgs;
      if (failed(parser.parseArgumentList(
              thenArgs, AsmParser::Delimiter::OptionalParen, true, false)))
        return failure();
      if (failed(parser.parseRegion(
              *elifRegionsRegions.emplace_back(std::make_unique<Region>()),
              thenArgs)))
        return failure();
      continue;
    }

    elseRegion.takeBody(*region);
    return success();
  }
}

static void printIf(OpAsmPrinter &printer, Operation *ifOp, Region &thenRegion,
                    MutableArrayRef<Region> conditionalRegions,
                    Region &elseRegion) {
  auto printArgumentList = [&](ArrayRef<BlockArgument> args) {
    if (args.empty())
      return;
    printer << "(";
    printer.printRegionArgument(args.front());
    for (BlockArgument arg : args.slice(1)) {
      printer << ", ";
      printer.printRegionArgument(arg);
    }
    printer << ")";
  };

  printer.printRegion(thenRegion);

  assert(conditionalRegions.size() % 2 == 0);
  for (unsigned i = 0, e = conditionalRegions.size(); i != e; i += 2) {
    printer << " else ";
    printArgumentList(conditionalRegions[i].getArguments());
    printer.printRegion(conditionalRegions[i], /*printEntryBlockArgs=*/false);
    printer << " then ";
    printArgumentList(conditionalRegions[i + 1].getArguments());
    printer.printRegion(conditionalRegions[i + 1],
                        /*printEntryBlockArgs=*/false);
  }
  printer << " else ";
  printArgumentList(elseRegion.getArguments());
  printer.printRegion(elseRegion, /*printEntryBlockArgs=*/false);
}

LogicalResult IfOp::verify() {
  if (getElifRegions().size() % 2 != 0) {
    return emitOpError(
        "operator elif conditions do not match the number of elif regions.");
  }
  return success();
}

void IfOp::getEntryTargets(ArrayRef<Attribute> operands,
                           SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  // Region layout: 0 = then, 1 = else, 2+ = additional (cond, then) pairs.
  unsigned nextOnFalse = getElifRegions().empty() ? 1 : 2;
  if (auto cond = dyn_cast_if_present<KGEN::SIMDAttr>(operands.front())) {
    targets.emplace_back(cond.getAsBool() ? 0 : nextOnFalse);
  } else {
    targets.emplace_back(0);
    targets.emplace_back(nextOnFalse);
  }
}

ValueRange IfOp::getEntryArguments(std::optional<unsigned int> target) {
  if (!target)
    return getResults();
  assert(*target < getNumRegions());
  return getRegion(target.value()).getArguments();
}

ErrorTreeOrSuccess IfOp::interpret(ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  auto cond = dyn_cast_if_present<KGEN::SIMDAttr>(operands[0]);
  if (!cond)
    return ErrorTree(getLoc(), "non-constant condition");

  if (cond.getAsBool())
    return state.transferControlFlowTo(getThenRegion(), {});
  if (!getElifRegions().empty())
    return state.transferControlFlowTo(getElifRegions()[0], {});
  return state.transferControlFlowTo(getElseRegion(), {});
}

ErrorTreeOrSuccess
IfOp::parametric_interpret(ArrayRef<Attribute> operands,
                           ParametricInterpreterState &state) {
  return interpret(operands, state);
}

IfOp IfOp::create(OpBuilder &builder, Location loc, TypeRange resultTypes,
                  Value cond, function_ref<LogicalResult()> emitThen,
                  function_ref<LogicalResult()> emitElse) {
  IfOp ifOp = IfOp::create(builder, loc, resultTypes, cond);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
  if (failed(emitThen()))
    return {};

  builder.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
  if (failed(emitElse()))
    return {};

  builder.setInsertionPointAfter(ifOp);
  return ifOp;
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
// IfElifCondYieldOp
//===----------------------------------------------------------------------===//

bool IfElifCondYieldOp::isParentNode(Operation *op) { return isa<IfOp>(op); }

void IfElifCondYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  // The first operand is the condition and the subsequent operands are likely
  // stack values that were promoted to register values and thus now rely on
  // block arguments.
  assert(!operands.empty());
  // Region layout: 0 = then, 1 = else, 2+ = elifRegions.
  unsigned myIndex = getOperation()->getParentRegion()->getRegionNumber();
  assert(myIndex >= 2 &&
         "if.elifcond.yield only belongs in additional cond regions");
  unsigned nextValueRegion = myIndex + 1;
  unsigned nextConditionRegion = myIndex + 2;
  unsigned numRegions =
      getOperation()->getParentRegion()->getParentOp()->getNumRegions();
  // Fall to else (region 1) when there is no next condition region.
  unsigned nextConditionRegionOrElse =
      nextConditionRegion >= numRegions ? 1 : nextConditionRegion;
  ValueRange carryOver(getOperands().drop_front(1));
  if (auto constantResult =
          dyn_cast_if_present<KGEN::SIMDAttr>(operands.front())) {
    targets.emplace_back(ControlFlowTarget(constantResult.getAsBool()
                                               ? nextValueRegion
                                               : nextConditionRegionOrElse,
                                           carryOver));
    return;
  }
  targets.emplace_back(nextValueRegion, carryOver);
  targets.emplace_back(nextConditionRegionOrElse, carryOver);
}

ErrorTreeOrSuccess IfElifCondYieldOp::interpret(ArrayRef<Attribute> operands,
                                                InterpreterState &state) {
  auto parent = cast<IfOp>(getOperation()->getParentOp());
  // Region layout: 0 = then, 1 = else, 2+ = elifRegions.
  unsigned myRegionNumber =
      getOperation()->getParentRegion()->getRegionNumber();
  assert(myRegionNumber >= 2);
  unsigned myElifIndex = myRegionNumber - 2;
  ArrayRef<Attribute> blockArguments = operands.slice(1);
  if (auto cond = dyn_cast_if_present<KGEN::SIMDAttr>(operands[0])) {
    if (cond.getAsBool()) {
      return state.transferControlFlowTo(
          parent.getElifRegions()[myElifIndex + 1], blockArguments);
    }
    unsigned nextIndex = myElifIndex + 2;
    if (nextIndex < parent.getElifRegions().size()) {
      return state.transferControlFlowTo(parent.getElifRegions()[nextIndex],
                                         blockArguments);
    }
    return state.transferControlFlowTo(parent.getElseRegion(), blockArguments);
  }
  return ErrorTree(getLoc(), "non-constant condition in elif chain.");
}

ErrorTreeOrSuccess
IfElifCondYieldOp::parametric_interpret(ArrayRef<Attribute> operands,
                                        ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

static ParseResult
parseMatch(OpAsmParser &parser,
           SmallVectorImpl<std::unique_ptr<Region>> &caseRegions,
           Region &elseRegion) {
  // First case region (required).
  if (failed(parser.parseRegion(
          *caseRegions.emplace_back(std::make_unique<Region>()))))
    return failure();

  // Additional case regions.
  while (succeeded(parser.parseOptionalKeyword("case"))) {
    if (failed(parser.parseRegion(
            *caseRegions.emplace_back(std::make_unique<Region>()))))
      return failure();
  }

  if (failed(parser.parseKeyword("else")) ||
      failed(parser.parseRegion(elseRegion)))
    return failure();
  return success();
}

static void printMatch(OpAsmPrinter &printer, Operation *op,
                       MutableArrayRef<Region> caseRegions,
                       Region &elseRegion) {
  assert(!caseRegions.empty() && "match requires at least one case region");
  printer.printRegion(caseRegions.front());
  for (Region &region : caseRegions.drop_front()) {
    printer.printNewline();
    printer << "case ";
    printer.printRegion(region);
  }
  printer << " else ";
  printer.printRegion(elseRegion);
}

bool MatchOp::containsInCaseRegion(Operation *op) {
  return getCaseRegionIndexContaining(op).has_value();
}

std::optional<unsigned> MatchOp::getCaseRegionIndexContaining(Operation *op) {
  Region *elseRegion = &getElseRegion();
  for (Region *r = op->getParentRegion(); r; r = r->getParentRegion()) {
    if (r == elseRegion)
      return std::nullopt;
    if (r->getParentOp() == getOperation()) {
      // Region 0 is else; case regions start at 1.
      assert(r->getRegionNumber() >= 1 && "expected a case region");
      return r->getRegionNumber() - 1;
    }
  }
  return std::nullopt;
}

LogicalResult MatchOp::verify() {
  if (getCaseRegions().empty())
    return emitOpError("requires at least one case region");
  return success();
}

void MatchOp::getEntryTargets(ArrayRef<Attribute> operands,
                              SmallVectorImpl<ControlFlowTarget> &targets) {
  (void)operands;
  // Begin in the first case region (region #1; #0 is else).
  targets.emplace_back(1);
}

ValueRange MatchOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target < getNumRegions());
  return getRegion(*target).getArguments();
}

ErrorTreeOrSuccess MatchOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  (void)operands;
  return state.transferControlFlowTo(getCaseRegions().front(), {});
}

ErrorTreeOrSuccess
MatchOp::parametric_interpret(ArrayRef<Attribute> operands,
                              ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// ComptimeMatchOp
//===----------------------------------------------------------------------===//

bool ComptimeMatchOp::containsInCaseRegion(Operation *op) {
  return getCaseRegionIndexContaining(op).has_value();
}

std::optional<unsigned>
ComptimeMatchOp::getCaseRegionIndexContaining(Operation *op) {
  Region *elseRegion = &getElseRegion();
  for (Region *r = op->getParentRegion(); r; r = r->getParentRegion()) {
    if (r == elseRegion)
      return std::nullopt;
    if (r->getParentOp() == getOperation()) {
      // Region 0 is else; case regions start at 1.
      assert(r->getRegionNumber() >= 1 && "expected a case region");
      return r->getRegionNumber() - 1;
    }
  }
  return std::nullopt;
}

LogicalResult ComptimeMatchOp::verify() {
  if (getCaseRegions().empty())
    return emitOpError("requires at least one case region");
  return success();
}

void ComptimeMatchOp::getEntryTargets(
    ArrayRef<Attribute> operands, SmallVectorImpl<ControlFlowTarget> &targets) {
  (void)operands;
  // Begin in the first case region (region #1; #0 is else).
  targets.emplace_back(1);
}

ValueRange ComptimeMatchOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target < getNumRegions());
  return getRegion(*target).getArguments();
}

ErrorTreeOrSuccess ComptimeMatchOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  (void)operands;
  return state.transferControlFlowTo(getCaseRegions().front(), {});
}

ErrorTreeOrSuccess
ComptimeMatchOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// MatchNextOp
//===----------------------------------------------------------------------===//

bool MatchNextOp::isParentNode(Operation *op) {
  if (auto match = dyn_cast<MatchOp>(op))
    return match.containsInCaseRegion(*this);
  if (auto match = dyn_cast<ComptimeMatchOp>(op))
    return match.containsInCaseRegion(*this);
  return false;
}

void MatchNextOp::getBranchTargets(
    ArrayRef<Attribute> operands, SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  auto emitTargets = [&](auto match) {
    std::optional<unsigned> caseIdx = match.getCaseRegionIndexContaining(*this);
    assert(caseIdx && "match.next must be nested in a case region");
    if (*caseIdx + 1 < match.getCaseRegions().size())
      // Next case region number is caseIdx+1 + 1 (else is region 0).
      targets.emplace_back(*caseIdx + 2, getOperands());
    else
      targets.emplace_back(0, getOperands()); // else region
  };
  Operation *parent = getParentNode(*this);
  if (auto match = dyn_cast<MatchOp>(parent))
    return emitTargets(match);
  emitTargets(cast<ComptimeMatchOp>(parent));
}

ErrorTreeOrSuccess MatchNextOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  auto transfer = [&](auto match) -> ErrorTreeOrSuccess {
    std::optional<unsigned> caseIdx = match.getCaseRegionIndexContaining(*this);
    assert(caseIdx && "match.next must be nested in a case region");
    if (*caseIdx + 1 < match.getCaseRegions().size())
      return state.transferControlFlowTo(match.getCaseRegions()[*caseIdx + 1],
                                         operands);
    return state.transferControlFlowTo(match.getElseRegion(), operands);
  };
  Operation *parent = getParentNode(*this);
  if (auto match = dyn_cast<MatchOp>(parent))
    return transfer(match);
  return transfer(cast<ComptimeMatchOp>(parent));
}

ErrorTreeOrSuccess
MatchNextOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// MatchCompleteOp
//===----------------------------------------------------------------------===//

bool MatchCompleteOp::isParentNode(Operation *op) {
  if (auto match = dyn_cast<MatchOp>(op))
    return match.containsInCaseRegion(*this);
  if (auto match = dyn_cast<ComptimeMatchOp>(op))
    return match.containsInCaseRegion(*this);
  return false;
}

void MatchCompleteOp::getBranchTargets(
    ArrayRef<Attribute> operands, SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(std::nullopt, getOperands());
}

ErrorTreeOrSuccess MatchCompleteOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  return state.transferControlFlowTo(getParentNode(*this), operands);
}

ErrorTreeOrSuccess
MatchCompleteOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// ComptimeForOp
//===----------------------------------------------------------------------===//

LogicalResult ComptimeForOp::verify() {
  if (getNumOperands() != getNumResults()) {
    return emitOpError("has ")
           << getNumOperands() << " operands but " << getNumResults()
           << " results; it should be the same";
  }
  for (auto [i, argTy, resTy] :
       llvm::enumerate(getOperandTypes(), getResultTypes())) {
    if (argTy == resTy)
      continue;
    return emitOpError("operand #")
           << i << " has type " << argTy
           << " but corresponding result has type " << resTy;
  }
  return success();
}

void ComptimeForOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(0, getOperands());
}

ValueRange ComptimeForOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  if (*target == 0)
    return getBody().getArguments();
  assert(*target == 1);
  return getElseRegion().getArguments();
}

ArrayRef<ParamDeclAttr> ComptimeForOp::getInputParams() {
  // DeclInterface requires ArrayRef; point at the inherent property storage
  // (not getAttrs().back(), which is wrong once attrs live in Properties).
  return {&getProperties().paramDecl, 1};
}

void ComptimeForOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

bool ComptimeForOp::isImplicitlyParametric() { return true; }

void ComptimeForOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

bool ComptimeForOp::isIsolatedFromAbove(unsigned regionNum) {
  if (regionNum == 0)
    return getBodyIsolated();
  assert(regionNum == 1);
  return getElseIsolated();
}

void ComptimeForOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  if (regionNum == 0)
    return setBodyIsolated(true);
  assert(regionNum == 1);
  return setElseIsolated(true);
}

bool ComptimeForBreakOp::isParentNode(Operation *op) {
  return isa<ComptimeForOp>(op);
}

void ComptimeForBreakOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the loop operation.
  targets.emplace_back(std::nullopt, getOperands());
}

bool ComptimeForContinueOp::isParentNode(Operation *op) {
  return isa<ComptimeForOp>(op);
}

void ComptimeForContinueOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to the beginning of the body region only (not the else region).
  targets.emplace_back(0, getOperands());
}

bool ComptimeForGotoElseOp::isParentNode(Operation *op) {
  return isa<ComptimeForOp>(op);
}

void ComptimeForGotoElseOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty() && "Shouldn't exist by mem2reg time");
  // Branch to the beginning of the else region.
  targets.emplace_back(1, ValueRange());
}

//===----------------------------------------------------------------------===//
// ComptimeIfOp
//===----------------------------------------------------------------------===//

bool ComptimeIfOp::isIsolatedFromAbove(unsigned regionNum) {
  switch (regionNum) {
  case 0:
    return getThenIsolated();
  case 1:
    return getElseIsolated();
  default:
    llvm_unreachable("unknown region number");
  }
}

void ComptimeIfOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  switch (regionNum) {
  case 0:
    setThenIsolated(true);
    break;
  case 1:
    setElseIsolated(true);
    break;
  default:
    llvm_unreachable("unknown region number");
  }
}

void ComptimeIfOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty());
  targets.emplace_back(0);
  targets.emplace_back(1);
}

ValueRange ComptimeIfOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0 || *target == 1);
  return {};
}

void ComptimeIfOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

bool ComptimeIfOp::isImplicitlyParametric() { return true; }

/// This operation has no uses to collect in the scopes it defines.
void ComptimeIfOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

//===----------------------------------------------------------------------===//
// ComptimeYieldOp
//===----------------------------------------------------------------------===//

bool ComptimeYieldOp::isParentNode(Operation *op) {
  return isa<ComptimeForOp, ComptimeIfOp>(op);
}

void ComptimeYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the if operation.
  targets.emplace_back(std::nullopt, getOperands());
}

//===----------------------------------------------------------------------===//
// ComptimeForOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ComptimeForOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  llvm_unreachable("hlcf.comptime.for interpret undefined");
}

ErrorTreeOrSuccess
ComptimeForOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  SmallVector<Type> resultTypes;
  Attribute hasNext = state.getReboundAttribute(getHasNext());
  Attribute getNext = state.getReboundAttribute(getGetNextIter());
  for (Type type : getResultTypes()) {
    resultTypes.push_back(state.getReboundType(type));
  }

  auto hasNextCall = cast<SymbolConstantAttr>(hasNext);
  auto iter = state.currOpSideEffectState().find(this->getOperation());
  bool firstIteration =
      (iter == state.currOpSideEffectState().end() || !iter->second.iterator);

  // Can probably cache this for each iteration.
  SmallVector<TypedAttr> paramValues;
  for (auto pv : hasNextCall.getParamValues()) {
    paramValues.push_back(state.getReboundAttribute(pv));
  }

  ErrorOr<Type> hasNextTypeResult =
      state.lookupFuncTypeGenerator(hasNextCall.getSymbol());
  if (hasNextTypeResult.isError()) {
    return ErrorTree(getLoc(), hasNextTypeResult.takeError());
  }

  FuncType hasNextType =
      cast<FuncTypeGeneratorType>(*hasNextTypeResult).getBody();

  // Push an empty slot to paramValues count to mark this is the boundary
  // of a ComptimeFor so that we know how much to pop once hitting
  // hlcf.comptime.for.break or hlcf.comptime.for.continue
  // state.pushParamValues({}, false, this->getOperation());
  Attribute initial = state.getReboundAttribute(getInitial());
  TypedAttr iterator =
      cast<TypedAttr>(firstIteration ? initial : iter->second.iterator);

  TypedAttr hasNextInput = iterator;
  if (hasAddress(hasNextType.getArgConvention(0)))
    hasNextInput = StoreToMemAttr::get(iterator, hasNextType.getArguments()[0]);

  ErrorTreeOr<TypedAttr> hasNextResult =
      state.interpretGenerator(hasNextCall, paramValues, iterator, getLoc());
  if (hasNextResult.isError()) {
    return hasNextResult.takeError();
  }

  if (!cast<BoolAttr>(*hasNextResult).getValue()) {
    // Go to else region
    ArrayRef<Attribute> arguments =
        firstIteration ? operands : iter->second.operands;
    state.currOpSideEffectState().erase(this->getOperation());
    (void)state.transferControlFlowTo(this->getOperation(), arguments);

  } else {
    state.overwriteDeclBinding(getParamDecl(), iterator);
    iterator =
        StoreToMemAttr::get(iterator, PointerType::get(iterator.getType()));

    auto getNextCall = cast<SymbolConstantAttr>(getNext);
    paramValues.clear();
    for (auto pv : getNextCall.getParamValues()) {
      paramValues.push_back(state.getReboundAttribute(pv));
    }

    ErrorTreeOr<TypedAttr> getNextResult =
        state.interpretGeneratorWithResultSlot(getNextCall, paramValues,
                                               iterator, getLoc());
    if (getNextResult.isError())
      return getNextResult.takeError();

    if (firstIteration) {
      state.currOpSideEffectState()[this->getOperation()] = {
          {}, {}, *getNextResult};
    } else {
      // Clear up iterator in case function returns in the body of the
      // ComptimeFor so that the iterator value doesn't carry over to another
      // round of interpreting this ComptimeFor by mistake.
      iter->second.iterator = {};
      // Set nextIterator value so that hlcf.comptime.for.continue can set the
      // iterator value correctly for the next iteration.
      iter->second.nextIterator = *getNextResult;
    }

    ArrayRef<Attribute> arguments =
        firstIteration ? operands : iter->second.operands;

    state.pushParamValues({iterator}, false, this->getOperation());
    state.pushEvalFrame(getOperation(), &getBody(), {}, 5);
    return state.transferControlFlowTo(getBody(), arguments);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ComptimeForBreakOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ComptimeForBreakOp::interpret(ArrayRef<Attribute> operands,
                                                 InterpreterState &state) {
  llvm_unreachable("hlcf.comptime.for.break interpret undefined");
}

ErrorTreeOrSuccess
ComptimeForBreakOp::parametric_interpret(ArrayRef<Attribute> operands,
                                         ParametricInterpreterState &state) {
  auto parent = this->getOperation()->getParentOfType<ComptimeForOp>();
  state.popEvalFrame();
  state.popParamValues(false, this->getOperation(), parent);
  return state.transferControlFlowTo(parent, operands);
}

//===----------------------------------------------------------------------===//
// ComptimeForContinueOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
ComptimeForContinueOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state) {
  llvm_unreachable("hlcf.comptime.for.continue interpret undefined");
}

ErrorTreeOrSuccess
ComptimeForContinueOp::parametric_interpret(ArrayRef<Attribute> operands,
                                            ParametricInterpreterState &state) {
  if (auto parent =
          this->getOperation()->getParentOfType<HLCF::ComptimeForOp>()) {
    state.popEvalFrame();
    state.popParamValues(false, this->getOperation(), parent);
    (void)state.transferControlFlowToParent(parent, operands);
    auto iter = state.currOpSideEffectState().find(parent.getOperation());
    assert(iter != state.currOpSideEffectState().end() &&
           "hlcf.comptime.for.continue has broken state");
    iter->second.operands = SmallVector<Attribute>(operands);
    iter->second.iterator = iter->second.nextIterator;
    return success();
  }
  return ErrorTree(getLoc(),
                   "INTERNAL ERROR: cannot find parent ComptimeForOp");
}

//===----------------------------------------------------------------------===
// ComptimeIfOp
//===----------------------------------------------------------------------===

ErrorTreeOrSuccess ComptimeIfOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  llvm_unreachable("hlcf.comptime.if interpret undefined");
}

ErrorTreeOrSuccess
ComptimeIfOp::parametric_interpret(ArrayRef<Attribute> operands,
                                   ParametricInterpreterState &state) {
  Attribute cond = state.getReboundAttribute(getCond());
  unsigned regionId = 2;
  if (auto result = sugarDynCast<SIMDAttr>(cond)) {
    regionId = result.getAsBool() ? 0 : 1;
  }

  if (regionId < 2) {
    Region &target = getRegion(regionId);
    state.pushParamValues({}, false);
    state.pushEvalFrame(getOperation(), &target, {}, 6);
    return state.transferControlFlowTo(target, {});
  }

  return ErrorTree(getLoc(), "wrong param if condition");
}

//===----------------------------------------------------------------------===//
// ComptimeYieldOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ComptimeYieldOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  llvm_unreachable("hlcf.comptime.yield interpret undefined");
}

ErrorTreeOrSuccess
ComptimeYieldOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  state.popEvalFrame();
  state.popParamValues(false, this->getOperation());
  return state.transferControlFlowTo((*this)->getParentOp(), operands);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/HLCFDialect/HLCF.cpp.inc"
