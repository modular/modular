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

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

// The declarative formats name these unqualified.
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"

//===----------------------------------------------------------------------===//
// map.call.opaque
//===----------------------------------------------------------------------===//

LogicalResult
CallOpaqueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  return mlir::success();
}

LogicalResult CallOpaqueOp::verify() {
  // `$outputs` is `AnyType`; a by-value `!mo.opaque` belongs in
  // `$opaqueResult`.
  for (Value output : getOutputs())
    if (isa<MO::OpaqueType>(output.getType()))
      return emitOpError("an opaque is returned through its result, not passed "
                         "as a destination in 'outputs'");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.call.elementwise
//===----------------------------------------------------------------------===//

LogicalResult
CallElementwiseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  if (!decl.getElementwise())
    return emitOpError("kernel '")
           << getCallee() << "' has no 'elementwise' function";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Shared parse/print helpers
//===----------------------------------------------------------------------===//

// Prints `%v : type, ...` (no surrounding parens).
static void printValuesWithTypes(OpAsmPrinter &p, ValueRange values) {
  llvm::interleaveComma(values, p,
                        [&](Value v) { p << v << " : " << v.getType(); });
}

//===----------------------------------------------------------------------===//
// map.iter.opaque
//===----------------------------------------------------------------------===//

ParseResult IterOpaqueOp::parse(OpAsmParser &parser, OperationState &result) {
  MOGG::DispatchKindAttr dispatch;
  if (MOGG::parseOptionalDispatch(parser, dispatch))
    return mlir::failure();
  if (dispatch)
    result.addAttribute(getDispatchAttrName(result.name), dispatch);

  // inputs ( %A : tA, ... ), a trailing chain included; the chain is split
  // out into `$inputChain` below.
  SmallVector<OpAsmParser::Argument> allArgInfo;
  if (parser.parseKeyword("inputs") ||
      parseParenOperandListWithShadowing(parser, result, allArgInfo,
                                         /*isIsolatedFromAbove=*/true))
    return mlir::failure();
  bool hasInputChain =
      !allArgInfo.empty() && isa<MO::ChainType>(allArgInfo.back().type);
  int32_t numInputOperands =
      static_cast<int32_t>(result.operands.size()) - (hasInputChain ? 1 : 0);
  // The chain has no body block argument.
  if (hasInputChain)
    allArgInfo.pop_back();

  // outputs ( %out : tOut, ... ); a trailing chain is a result without a
  // block argument.
  SmallVector<OpAsmParser::Argument> outArgs;
  if (parser.parseKeyword("outputs") ||
      parser.parseArgumentList(outArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return mlir::failure();
  // `returns tOpaque`: a by-value result with no block argument.
  Type opaqueResultType;
  if (succeeded(parser.parseOptionalKeyword("returns")) &&
      parser.parseType(opaqueResultType))
    return mlir::failure();

  // `$results` is ordered [DPS results..., by-value opaque?, chain?],
  // regardless of the written order.
  Type chainResultType;
  for (const OpAsmParser::Argument &arg : outArgs) {
    if (isa<MO::ChainType>(arg.type)) {
      chainResultType = arg.type;
      continue;
    }
    result.addTypes(arg.type);
    allArgInfo.push_back(arg);
  }
  if (opaqueResultType)
    result.addTypes(opaqueResultType);
  if (chainResultType)
    result.addTypes(chainResultType);

  result.addAttribute(IterOpaqueOp::getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {numInputOperands, hasInputChain ? 1 : 0}));

  Region *body = result.addRegion();
  if (parseRegion(parser, allArgInfo, *body, /*isIsolatedFromAbove=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

void IterOpaqueOp::print(OpAsmPrinter &p) {
  if (getDispatch() != MOGG::DispatchKind::Deferred) {
    p << " ";
    MOGG::printOptionalDispatch(p, getDispatchAttr(),
                                MOGG::DispatchKind::Deferred);
  }
  p.increaseIndent();
  p.printNewline();
  p << "inputs ";
  // Inlines `printParenOperandListWithShadowing`, which needs a real
  // `OperandRange`, then appends `$inputChain` as a trailing `%chain: type`.
  ValueRange inputs = getInputs();
  Block::BlockArgListType inputArgs = getInputBlockArgs();
  {
    llvm::DenseSet<Value> seenInputs;
    bool allDistinctInputs = true;
    for (Value v : inputs)
      if (!seenInputs.insert(v).second)
        allDistinctInputs = false;
    bool needAs = !allDistinctInputs;
    p << "(";
    for (auto [i, pair] : llvm::enumerate(llvm::zip(inputs, inputArgs))) {
      auto [operand, arg] = pair;
      if (i != 0)
        p << ", ";
      p << operand;
      if (needAs)
        p << " as " << arg;
      p << ": " << operand.getType();
    }
    if (Value inputChain = getInputChain()) {
      if (!inputs.empty())
        p << ", ";
      p << inputChain << ": " << inputChain.getType();
    }
    p << ")";
  }
  p.printNewline();
  p << "outputs (";
  printValuesWithTypes(p, getOutputBlockArgs());
  if (Value outputChain = getOutputChain()) {
    if (getNumNonChainResults() != 0)
      p << ", ";
    p << outputChain << ": " << outputChain.getType();
  }
  p << ") ";
  if (Value opaqueResult = getOpaqueResult())
    p << "returns " << opaqueResult.getType() << " ";
  p.decreaseIndent();

  // Inner references to the inputs print with the outer operand's SSA name.
  {
    llvm::DenseSet<Value> seen;
    bool inputsDistinct = true;
    for (Value v : inputs) {
      if (!seen.insert(v).second) {
        inputsDistinct = false;
        break;
      }
    }
    if (inputsDistinct) {
      SmallVector<Value> shadowValues(inputs.begin(), inputs.end());
      for (BlockArgument arg : getOutputBlockArgs())
        shadowValues.push_back(arg);
      p.shadowRegionArgs(getBody(), shadowValues);
    }
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getOperandSegmentSizesAttrName().getValue(),
                           getDispatchAttrName().getValue()});
}

LogicalResult IterOpaqueOp::verify() {
  // The block-arg accessors and the epilogue fusers (result `i` <-> output
  // block arg `i`) rely on [DPS results..., by-value opaque?, chain?]; any
  // other order would silently misalign them.
  bool seenByValue = false;
  for (Value result : getResults()) {
    Type ty = result.getType();
    if (isa<MO::OpaqueType>(ty) || isa<MO::ChainType>(ty)) {
      seenByValue = true;
      continue;
    }
    if (seenByValue)
      return emitOpError("destination-passing results must precede a by-value "
                         "opaque result and the output chain");
  }

  // A by-value result needs a `map.yield` of something other than a DPS
  // output block arg (yielding only those is legal but redundant).
  if (getOpaqueResult()) {
    auto yieldOp = dyn_cast<YieldOp>(getBody().front().back());
    Block::BlockArgListType outputArgs = getOutputBlockArgs();
    bool yieldsByValue =
        yieldOp && llvm::any_of(yieldOp.getOperands(), [&](Value operand) {
          return !llvm::is_contained(outputArgs, operand);
        });
    if (!yieldsByValue)
      return emitOpError("has a by-value result but its body does not "
                         "map.yield a value for it");
  }

  return mlir::success();
}

Value IterOpaqueOp::getOpaqueResult() {
  for (Value result : getResults())
    if (isa<MO::OpaqueType>(result.getType()))
      return result;
  return {};
}

unsigned IterOpaqueOp::getNumNonChainResults() {
  return getNumResults() - (getOutputChain() ? 1 : 0) -
         (getOpaqueResult() ? 1 : 0);
}

Block::BlockArgListType IterOpaqueOp::getOutputBlockArgs() {
  return getBody().front().getArguments().take_back(getNumNonChainResults());
}

Block::BlockArgListType IterOpaqueOp::getInputBlockArgs() {
  return getBody().front().getArguments().drop_back(getNumNonChainResults());
}

Value IterOpaqueOp::getInputFor(BlockArgument arg) {
  ValueRange inputs = getInputs();
  Block::BlockArgListType inputArgs = getInputBlockArgs();
  for (unsigned i = 0, e = inputs.size(); i < e; ++i)
    if (inputArgs[i] == arg)
      return inputs[i];
  return {};
}

IterEpilogueOp IterOpaqueOp::getEpilogueForResult(unsigned resultIdx) {
  Value outArg = getOutputBlockArgs()[resultIdx];
  for (auto epilogueOp : getBody().front().getOps<IterEpilogueOp>())
    if (llvm::is_contained(epilogueOp.getCaptures(), outArg))
      return epilogueOp;
  return nullptr;
}

void IterOpaqueOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().front().insertArgument(idx, v.getType(), v.getLoc());
}

void IterOpaqueOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(), << "[idx, idx + len) = ["
                                                 << idx << ", " << (idx + len)
                                                 << "), before outputs");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().front().eraseArgument(idx);
}

Value IterOpaqueOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

void IterOpaqueOp::setChain(Value chain) {
  if (chain)
    getInputChainMutable().assign(chain);
}

// Reports hasReadEffect/hasWriteEffect per mutable operand, falling back to an
// unattributed write when no operand is mutable.
static void
collectKernelEffects(Operation *op, bool hasReadEffect, bool hasWriteEffect,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  bool addedEffect = false;
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (hasWriteEffect)
      effects.emplace_back(MemoryEffects::Write::get(), &opOperand);
    if (hasReadEffect)
      effects.emplace_back(MemoryEffects::Read::get(), &opOperand);
    addedEffect = true;
  }

  if (!addedEffect && (hasReadEffect || hasWriteEffect))
    effects.emplace_back(MemoryEffects::Write::get());
}

void IterOpaqueOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectKernelEffects(getOperation(), getHasReadEffect(), getHasWriteEffect(),
                       effects);
}

void IterOpaqueOp::getAsmBlockArgumentNames(Region &region,
                                            OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = getOutputBlockArgs();
  for (auto [i, arg] : llvm::enumerate(args))
    setNameFn(arg, args.size() == 1 ? "out" : ("out" + std::to_string(i)));
}

//===----------------------------------------------------------------------===//
// map.iter.epilogue
//===----------------------------------------------------------------------===//

ParseResult IterEpilogueOp::parse(OpAsmParser &parser, OperationState &result) {
  // Leading block args: ( %val : tVal, %idx : tIdx )
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return mlir::failure();

  // capturing {%inner = %outer : type, ...}: each %outer is an operand
  // shadowed by the block argument %inner.
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (M::MOGG::parseBindings(parser, "capturing", entryArgs, captures,
                             captureTypes, /*impliedType=*/Type(),
                             /*required=*/true))
    return mlir::failure();
  if (parser.resolveOperands(captures, captureTypes,
                             parser.getCurrentLocation(), result.operands))
    return mlir::failure();

  // Optional `intermediate %outer : type`, a trailing operand with no block
  // argument.
  int32_t numIntermediateTensor = 0;
  if (succeeded(parser.parseOptionalKeyword("intermediate"))) {
    OpAsmParser::UnresolvedOperand srcOperand;
    Type srcType;
    if (parser.parseOperand(srcOperand) || parser.parseColonType(srcType) ||
        parser.resolveOperand(srcOperand, srcType, result.operands))
      return mlir::failure();
    numIntermediateTensor = 1;
  }
  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(captures.size()), numIntermediateTensor}));

  // Entry args are [leading..., captures...]. Name shadowing is valid because
  // the region is IsolatedFromAbove.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return mlir::failure();

  // : result-type
  Type resultType;
  if (parser.parseColonType(resultType))
    return mlir::failure();
  result.addTypes(resultType);

  return parser.parseOptionalAttrDict(result.attributes);
}

void IterEpilogueOp::print(OpAsmPrinter &p) {
  Block &block = getBody().front();
  ValueRange captures = getCaptures();
  unsigned numLeading = block.getNumArguments() - captures.size();

  p << " (";
  printValuesWithTypes(p, block.getArguments().take_front(numLeading));
  p << ") ";
  M::MOGG::printBindings(p, "capturing", captures,
                         block.getArguments().drop_front(numLeading),
                         /*impliedType=*/Type(), /*required=*/true);
  p << " ";
  if (Value intermediateTensor = getIntermediateTensor())
    p << "intermediate " << intermediateTensor << " : "
      << intermediateTensor.getType() << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getOperandSegmentSizesAttrName().getValue()});
}

void IterEpilogueOp::getAsmBlockArgumentNames(Region &region,
                                              OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  unsigned numCaptures = getCaptures().size();
  unsigned numLeading = args.size() - numCaptures;
  static constexpr const char *kLeadingNames[] = {"val", "idx"};
  for (auto [i, arg] : llvm::enumerate(args)) {
    if (i < numLeading) {
      setNameFn(arg, i < 2 ? kLeadingNames[i] : ("arg" + std::to_string(i)));
    } else {
      unsigned c = i - numLeading;
      setNameFn(arg, numCaptures == 1 ? "cap" : ("cap" + std::to_string(c)));
    }
  }
}

//===----------------------------------------------------------------------===//
// map.iter.prologue
//===----------------------------------------------------------------------===//

ParseResult IterPrologueOp::parse(OpAsmParser &parser, OperationState &result) {
  // Leading block arg: ( %idx : tIdx )
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  // capturing {%inner = %outer : type, ...}, as in map.iter.epilogue.
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (M::MOGG::parseBindings(parser, "capturing", entryArgs, captures,
                             captureTypes, /*impliedType=*/Type(),
                             /*required=*/true))
    return failure();
  if (parser.resolveOperands(captures, captureTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return failure();

  // : result-type
  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  return parser.parseOptionalAttrDict(result.attributes);
}

void IterPrologueOp::print(OpAsmPrinter &p) {
  Block &block = getBody().front();
  ValueRange captures = getCaptures();
  unsigned numLeading = block.getNumArguments() - captures.size();

  p << " (";
  printValuesWithTypes(p, block.getArguments().take_front(numLeading));
  p << ") ";
  M::MOGG::printBindings(p, "capturing", captures,
                         block.getArguments().drop_front(numLeading),
                         /*impliedType=*/Type(), /*required=*/true);
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

void IterPrologueOp::getAsmBlockArgumentNames(Region &region,
                                              OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  unsigned numCaptures = getCaptures().size();
  unsigned numLeading = args.size() - numCaptures;
  static constexpr const char *kLeadingNames[] = {"idx"};
  for (auto [i, arg] : llvm::enumerate(args)) {
    if (i < numLeading) {
      setNameFn(arg, i < 1 ? kLeadingNames[i] : ("arg" + std::to_string(i)));
    } else {
      unsigned c = i - numLeading;
      setNameFn(arg, numCaptures == 1 ? "cap" : ("cap" + std::to_string(c)));
    }
  }
}

//===----------------------------------------------------------------------===//
// map.iter.elementwise
//===----------------------------------------------------------------------===//

// Parses `(name: type, ...)` block arguments without operands.
static ParseResult
parseBlockArgList(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::Argument> &argInfo) {
  if (parser.parseLParen())
    return mlir::failure();
  if (succeeded(parser.parseOptionalRParen()))
    return mlir::success();
  do {
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/true))
      return mlir::failure();
    argInfo.push_back(arg);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

static void printBlockArgList(OpAsmPrinter &p, Block::BlockArgListType args) {
  p << "(";
  llvm::interleaveComma(args, p, [&](BlockArgument arg) {
    p.printOperand(arg);
    p << ": ";
    p.printType(arg.getType());
  });
  p << ")";
}

ParseResult IterElementwiseOp::parse(OpAsmParser &parser,
                                     OperationState &state) {
  // <[d0, d1, ...]>
  if (parser.parseLess())
    return mlir::failure();
  mlir::TypedAttr shapeTypedAttr;
  auto shapeResult = MOSH::parseShapeAttr(parser, shapeTypedAttr);
  if (!shapeResult.has_value())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '[' for map.iter.elementwise shape");
  if (failed(*shapeResult))
    return mlir::failure();
  if (parser.parseGreater())
    return mlir::failure();
  state.addAttribute(getShapeAttrName(state.name), shapeTypedAttr);

  MOGG::DispatchKindAttr dispatch;
  if (MOGG::parseOptionalDispatch(parser, dispatch))
    return mlir::failure();
  if (dispatch)
    state.addAttribute(getDispatchAttrName(state.name), dispatch);

  // inputs (%x: T, %y: T)
  SmallVector<OpAsmParser::Argument> allArgInfo;
  if (parser.parseKeyword("inputs") ||
      parseParenOperandListWithShadowing(parser, state, allArgInfo,
                                         /*isIsolatedFromAbove=*/true))
    return mlir::failure();
  int32_t numInputOperands = static_cast<int32_t>(state.operands.size());

  // outputs (%out: T), block args only.
  SmallVector<OpAsmParser::Argument> outputArgInfo;
  if (parser.parseKeyword("outputs") ||
      parseBlockArgList(parser, outputArgInfo))
    return mlir::failure();
  for (auto &arg : outputArgInfo)
    allArgInfo.push_back(arg);

  // index (%idx: map.index<N>), block arg only.
  SmallVector<OpAsmParser::Argument> indexArgInfo;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, indexArgInfo))
    return mlir::failure();
  if (indexArgInfo.size() != 1)
    return parser.emitError(
        parser.getCurrentLocation(),
        "'map.iter.elementwise' expects exactly one index argument");
  allArgInfo.push_back(indexArgInfo.front());

  // Optional chain (%c), an operand with no block arg.
  int32_t numChainOperands = 0;
  if (succeeded(parser.parseOptionalKeyword("chain"))) {
    OpAsmParser::UnresolvedOperand chainOperand;
    if (parser.parseLParen() || parser.parseOperand(chainOperand) ||
        parser.parseRParen())
      return mlir::failure();
    if (parser.resolveOperand(chainOperand,
                              MO::ChainType::get(parser.getContext()),
                              state.operands))
      return mlir::failure();
    numChainOperands = 1;
  }
  state.addAttribute(
      IterElementwiseOp::getOperandSegmentSizesAttrName(state.name),
      parser.getBuilder().getDenseI32ArrayAttr(
          {numInputOperands, numChainOperands}));

  if (parser.parseArrowTypeList(state.types))
    return mlir::failure();

  if (parseRegion(parser, allArgInfo, *state.addRegion(),
                  /*isIsolatedFromAbove=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterElementwiseOp::print(OpAsmPrinter &p) {
  // Without the outer "#mosh<ape[...]>" envelope.
  p << "<[";
  MOSH::printShapeValue(p, getShape().getValues());
  p << "]>";
  if (getDispatch() != MOGG::DispatchKind::Deferred) {
    p << " ";
    MOGG::printOptionalDispatch(p, getDispatchAttr(),
                                MOGG::DispatchKind::Deferred);
  }
  p << "\n    ";

  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();
  bool hasOutChain = getOutputChain() != nullptr;
  unsigned numOutputs = getResults().size() - (hasOutChain ? 1 : 0);

  p << "inputs  ";
  printParenOperandListWithShadowing(p, getInputs(),
                                     body.getArguments().slice(0, numInputs),
                                     /*isIsolatedFromAbove=*/true);
  p << "\n    ";

  p << "outputs ";
  printBlockArgList(p, body.getArguments().slice(numInputs, numOutputs));
  p << "\n    ";

  p << "index   ";
  printBlockArgList(p, body.getArguments().slice(numInputs + numOutputs, 1));
  p << " ";

  if (Value chain = getInputChain())
    p << "chain(" << chain << ") ";

  p.printArrowTypeList(getResultTypes());
  p << " ";

  // ParserUtils' printRegion() needs one operand per region argument, so
  // shadow manually; outputs and index shadow themselves.
  {
    llvm::DenseSet<Value> seen;
    bool inputsDistinct = true;
    for (Value v : getInputs()) {
      if (!seen.insert(v).second) {
        inputsDistinct = false;
        break;
      }
    }
    if (inputsDistinct) {
      SmallVector<Value> shadowValues(getInputs().begin(), getInputs().end());
      for (BlockArgument arg : body.getArguments().drop_front(numInputs))
        shadowValues.push_back(arg);
      p.shadowRegionArgs(getBody(), shadowValues);
    }
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {"shape", getOperandSegmentSizesAttrName().getValue(),
                           getDispatchAttrName().getValue()});
}

LogicalResult IterElementwiseOp::verify() {
  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();

  // A chain result must be last and has no block argument.
  ResultRange results = getResults();
  for (unsigned i = 0; i < results.size(); ++i) {
    bool isChain = llvm::isa<MO::ChainType>(results[i].getType());
    if (isChain && i + 1 != results.size())
      return emitOpError() << "only the last result may be a chain";
  }
  bool hasOutChain = getOutputChain() != nullptr;
  unsigned numOutputs = results.size() - (hasOutChain ? 1 : 0);
  unsigned expectedArgs = numInputs + numOutputs + 1; // +1 for index

  if (body.getNumArguments() != expectedArgs)
    return emitOpError() << "body expects " << expectedArgs
                         << " block arguments (" << numInputs << " inputs + "
                         << numOutputs << " outputs + 1 index), "
                         << "got " << body.getNumArguments();

  for (unsigned i = 0; i < numInputs; ++i) {
    if (body.getArgument(i).getType() != getInputs()[i].getType())
      return emitOpError() << "input block argument " << i
                           << " type mismatch with operand";
  }

  for (unsigned i = 0; i < numOutputs; ++i) {
    if (body.getArgument(numInputs + i).getType() != results[i].getType())
      return emitOpError() << "output block argument " << i
                           << " type mismatch with result";
  }

  BlockArgument indexArg = body.getArgument(numInputs + numOutputs);
  auto indexTy = llvm::dyn_cast<IndexType>(indexArg.getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";

  if (indexTy.getRank() != getShape().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match shape rank "
                         << getShape().getRank();

  return mlir::success();
}

Value IterElementwiseOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

void IterElementwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectKernelEffects(getOperation(), getHasReadEffect(), getHasWriteEffect(),
                       effects);
}

// The first device annotation on `results`, then `inputs`.
static llvm::SmallVector<M::DeviceRefAttr>
getFirstExecutionDevice(ValueRange results, ValueRange inputs) {
  for (ValueRange values : {results, inputs})
    for (Value v : values)
      if (auto dt =
              llvm::dyn_cast<M::MO::MOHasDeviceTypeInterface>(v.getType()))
        if (M::DeviceRefAttr device = dt.getDeviceRef())
          return {device};
  return {M::DeviceRefAttr{}};
}

llvm::SmallVector<M::DeviceRefAttr> IterElementwiseOp::getExecutionDevices() {
  return getFirstExecutionDevice(getResults(), getInputs());
}

Block::BlockArgListType IterElementwiseOp::getInputBlockArgs() {
  return getBody().front().getArguments().take_front(getInputs().size());
}

Block::BlockArgListType IterElementwiseOp::getOutputBlockArgs() {
  unsigned numOutputs = getNumResults() - (getOutputChain() ? 1 : 0);
  return getBody().front().getArguments().slice(getInputs().size(), numOutputs);
}

CapturingFusionClosure
IterElementwiseOp::createBodyClosure(OpBuilder &builder, Location loc,
                                     ValueRange captures) {
  return IterElementwiseBodyOp::create(builder, loc, getShape(), captures);
}

void IterElementwiseOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs/index block args");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().insertArgument(idx, v.getType(), v.getLoc());
}

void IterElementwiseOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(),
                << "[idx, idx + len) = [" << idx << ", " << (idx + len)
                << ") must stay within the inputs (" << getInputs().size()
                << "), before outputs/index block args");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().eraseArgument(idx);
}

void IterElementwiseOp::setChain(Value chain) {
  if (chain)
    getChainMutable().assign(chain);
}

//===----------------------------------------------------------------------===//
// map.iter.elementwise.body
//===----------------------------------------------------------------------===//

ParseResult IterElementwiseBodyOp::parse(OpAsmParser &parser,
                                         OperationState &state) {
  if (parser.parseLess())
    return mlir::failure();
  mlir::TypedAttr shapeTypedAttr;
  auto shapeResult = MOSH::parseShapeAttr(parser, shapeTypedAttr);
  if (!shapeResult.has_value())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '[' for map.iter.elementwise.body shape");
  if (failed(*shapeResult))
    return mlir::failure();
  if (parser.parseGreater())
    return mlir::failure();
  state.addAttribute(getShapeAttrName(state.name), shapeTypedAttr);

  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, entryArgs))
    return mlir::failure();
  if (entryArgs.size() != 1)
    return parser.emitError(
        parser.getCurrentLocation(),
        "'map.iter.elementwise.body' expects exactly one index argument");

  // capturing {%inner = %outer : type, ...}, as in map.iter.epilogue.
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (M::MOGG::parseBindings(parser, "capturing", entryArgs, captures,
                             captureTypes, /*impliedType=*/Type(),
                             /*required=*/true))
    return mlir::failure();
  if (parser.resolveOperands(captures, captureTypes,
                             parser.getCurrentLocation(), state.operands))
    return mlir::failure();

  Region *body = state.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterElementwiseBodyOp::print(OpAsmPrinter &p) {
  p << "<[";
  MOSH::printShapeValue(p, getShape().getValues());
  p << "]> ";

  Block &block = getBody().front();
  ValueRange captures = getCaptures();
  unsigned numLeading = block.getNumArguments() - captures.size();

  p.increaseIndent();
  p.printNewline();
  p << "index   ";
  printBlockArgList(p, block.getArguments().take_front(numLeading));
  p.printNewline();
  M::MOGG::printBindings(p, "capturing", captures,
                         block.getArguments().drop_front(numLeading),
                         /*impliedType=*/Type(), /*required=*/true);
  p << " ";
  p.decreaseIndent();

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"shape"});
}

LogicalResult IterElementwiseBodyOp::verify() {
  Block &body = getBody().front();
  unsigned numCaptures = getCaptures().size();
  if (body.getNumArguments() != 1 + numCaptures)
    return emitOpError() << "body expects exactly " << (1 + numCaptures)
                         << " block arguments (the index, then one per "
                            "capture), got "
                         << body.getNumArguments();

  auto indexTy = llvm::dyn_cast<IndexType>(body.getArgument(0).getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";

  if (indexTy.getRank() != getShape().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match shape rank "
                         << getShape().getRank();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.iter.rowwise
//===----------------------------------------------------------------------===//

// Parses `<[d0, d1, ...], axis = N>`.
static ParseResult parseRowwiseShapeAndAxis(OpAsmParser &parser,
                                            OperationState &state,
                                            StringAttr shapeName,
                                            StringAttr axisName) {
  if (parser.parseLess())
    return mlir::failure();
  mlir::TypedAttr shapeTypedAttr;
  auto shapeResult = MOSH::parseShapeAttr(parser, shapeTypedAttr);
  if (!shapeResult.has_value())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '[' for the rowwise shape");
  if (failed(*shapeResult))
    return mlir::failure();
  int64_t axis;
  if (parser.parseComma() || parser.parseKeyword("axis") ||
      parser.parseEqual() || parser.parseInteger(axis) || parser.parseGreater())
    return mlir::failure();
  state.addAttribute(shapeName, shapeTypedAttr);
  state.addAttribute(axisName, parser.getBuilder().getI64IntegerAttr(axis));
  return mlir::success();
}

static void printRowwiseShapeAndAxis(OpAsmPrinter &p, MOSH::ShapeAttr shape,
                                     int64_t axis) {
  p << "<[";
  MOSH::printShapeValue(p, shape.getValues());
  p << "], axis = " << axis << ">";
}

static LogicalResult verifyRowIndex(Operation *op, MOSH::ShapeAttr shape,
                                    int64_t axis, BlockArgument row) {
  int64_t rank = shape.getRank();
  if (axis < 0 || axis >= rank)
    return op->emitOpError()
           << "axis " << axis << " is out of range for rank " << rank;
  auto rowTy = llvm::dyn_cast<MAP::IndexType>(row.getType());
  if (!rowTy)
    return op->emitOpError() << "row block argument must be map.index<N>";
  if (static_cast<int64_t>(rowTy.getRank()) != rank)
    return op->emitOpError() << "row index rank " << rowTy.getRank()
                             << " does not match shape rank " << rank;
  return mlir::success();
}

ParseResult IterRowwiseOp::parse(OpAsmParser &parser, OperationState &state) {
  if (parseRowwiseShapeAndAxis(parser, state, getShapeAttrName(state.name),
                               getAxisAttrName(state.name)))
    return mlir::failure();

  MOGG::DispatchKindAttr dispatch;
  if (MOGG::parseOptionalDispatch(parser, dispatch))
    return mlir::failure();
  if (dispatch)
    state.addAttribute(getDispatchAttrName(state.name), dispatch);

  // inputs (%x: T, ...)
  SmallVector<OpAsmParser::Argument> allArgInfo;
  if (parser.parseKeyword("inputs") ||
      parseParenOperandListWithShadowing(parser, state, allArgInfo,
                                         /*isIsolatedFromAbove=*/true))
    return mlir::failure();
  int32_t numInputOperands = static_cast<int32_t>(state.operands.size());

  // outputs (%out: T, ...), block args only.
  if (parser.parseKeyword("outputs") || parseBlockArgList(parser, allArgInfo))
    return mlir::failure();

  // row (%row: map.index<N>), block arg only.
  SmallVector<OpAsmParser::Argument> rowArgInfo;
  if (parser.parseKeyword("row") || parseBlockArgList(parser, rowArgInfo))
    return mlir::failure();
  if (rowArgInfo.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "'map.iter.rowwise' expects exactly one row "
                            "argument");
  allArgInfo.push_back(rowArgInfo.front());

  // Optional chain (%c), an operand with no block arg.
  int32_t numChainOperands = 0;
  if (succeeded(parser.parseOptionalKeyword("chain"))) {
    OpAsmParser::UnresolvedOperand chainOperand;
    if (parser.parseLParen() || parser.parseOperand(chainOperand) ||
        parser.parseRParen() ||
        parser.resolveOperand(chainOperand,
                              MO::ChainType::get(parser.getContext()),
                              state.operands))
      return mlir::failure();
    numChainOperands = 1;
  }
  state.addAttribute(getOperandSegmentSizesAttrName(state.name),
                     parser.getBuilder().getDenseI32ArrayAttr(
                         {numInputOperands, numChainOperands}));

  if (parser.parseArrowTypeList(state.types))
    return mlir::failure();

  if (parseRegion(parser, allArgInfo, *state.addRegion(),
                  /*isIsolatedFromAbove=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterRowwiseOp::print(OpAsmPrinter &p) {
  printRowwiseShapeAndAxis(p, getShape(), getAxis());
  if (getDispatch() != MOGG::DispatchKind::Deferred) {
    p << " ";
    MOGG::printOptionalDispatch(p, getDispatchAttr(),
                                MOGG::DispatchKind::Deferred);
  }
  p << "\n    ";

  p << "inputs  ";
  printParenOperandListWithShadowing(p, getInputs(), getInputBlockArgs(),
                                     /*isIsolatedFromAbove=*/true);
  p << "\n    ";
  p << "outputs ";
  printBlockArgList(p, getOutputBlockArgs());
  p << "\n    ";
  p << "row     ";
  printBlockArgList(p, getBody().front().getArguments().take_back());
  if (Value chain = getInputChain())
    p << " chain(" << chain << ")";
  p.printArrowTypeList(getResultTypes());
  p << " ";

  // As in map.iter.elementwise: shadow manually, outputs and row shadow
  // themselves.
  llvm::DenseSet<Value> seen;
  if (llvm::all_of(getInputs(),
                   [&](Value v) { return seen.insert(v).second; })) {
    SmallVector<Value> shadowValues(getInputs().begin(), getInputs().end());
    for (BlockArgument arg :
         getBody().front().getArguments().drop_front(getInputs().size()))
      shadowValues.push_back(arg);
    p.shadowRegionArgs(getBody(), shadowValues);
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {getShapeAttrName().getValue(),
                           getAxisAttrName().getValue(),
                           getOperandSegmentSizesAttrName().getValue(),
                           getDispatchAttrName().getValue()});
}

LogicalResult IterRowwiseOp::verify() {
  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();

  // A chain result must be last and has no block argument.
  ResultRange results = getResults();
  for (auto [i, result] : llvm::enumerate(results))
    if (llvm::isa<MO::ChainType>(result.getType()) && i + 1 != results.size())
      return emitOpError() << "only the last result may be a chain";
  unsigned numOutputs = results.size() - (getOutputChain() ? 1 : 0);
  unsigned expectedArgs = numInputs + numOutputs + 1;
  if (body.getNumArguments() != expectedArgs)
    return emitOpError() << "body expects " << expectedArgs
                         << " block arguments (" << numInputs << " inputs + "
                         << numOutputs << " outputs + 1 row), got "
                         << body.getNumArguments();

  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getInputBlockArgs())))
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError() << "input block argument " << i
                           << " type mismatch with operand";
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(results, getOutputBlockArgs())))
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError() << "output block argument " << i
                           << " type mismatch with result";

  MOSH::ShapeAttr shape = getShape();
  int64_t axis = getAxis();
  if (failed(verifyRowIndex(*this, shape, axis, getRowArg())))
    return mlir::failure();

  // Each result has the full shape (per element) or the shape with `axis`
  // set to 1 (per row).
  for (auto [i, result] : llvm::enumerate(results.take_front(numOutputs))) {
    MOSH::ShapeAttr resultShape =
        llvm::cast<MO::TensorType>(result.getType()).getShape();
    bool perElement = resultShape == shape;
    bool perRow = resultShape.getRank() == shape.getRank() &&
                  MOSH::ShapeAttr::getStaticDim(resultShape[axis]) == 1;
    for (int64_t d = 0, e = shape.getRank(); perRow && d < e; ++d)
      perRow = d == axis || resultShape[d] == shape[d];
    if (!perElement && !perRow)
      return emitOpError() << "result " << i
                           << " must have the rowwise shape, or that shape "
                              "with dimension "
                           << axis << " set to 1; got " << result.getType();
  }
  return mlir::success();
}

Block::BlockArgListType IterRowwiseOp::getInputBlockArgs() {
  return getBody().front().getArguments().take_front(getInputs().size());
}

Block::BlockArgListType IterRowwiseOp::getOutputBlockArgs() {
  unsigned numOutputs = getNumResults() - (getOutputChain() ? 1 : 0);
  return getBody().front().getArguments().slice(getInputs().size(), numOutputs);
}

CapturingFusionClosure IterRowwiseOp::createBodyClosure(OpBuilder &builder,
                                                        Location loc,
                                                        ValueRange captures) {
  return IterRowwiseBodyOp::create(builder, loc, getShape(), getAxisAttr(),
                                   captures);
}

BlockArgument IterRowwiseOp::getRowArg() {
  return getBody().front().getArguments().back();
}

void IterRowwiseOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs/row block args");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().insertArgument(idx, v.getType(), v.getLoc());
}

void IterRowwiseOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(),
                << "[idx, idx + len) = [" << idx << ", " << (idx + len)
                << ") must stay within the inputs (" << getInputs().size()
                << "), before outputs/row block args");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().eraseArgument(idx);
}

llvm::SmallVector<M::DeviceRefAttr> IterRowwiseOp::getExecutionDevices() {
  return getFirstExecutionDevice(getResults(), getInputs());
}

Value IterRowwiseOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

void IterRowwiseOp::setChain(Value chain) {
  if (chain)
    getChainMutable().assign(chain);
}

void IterRowwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectKernelEffects(getOperation(), getHasReadEffect(), getHasWriteEffect(),
                       effects);
}

//===----------------------------------------------------------------------===//
// map.iter.rowwise.body
//===----------------------------------------------------------------------===//

ParseResult IterRowwiseBodyOp::parse(OpAsmParser &parser,
                                     OperationState &state) {
  if (parseRowwiseShapeAndAxis(parser, state, getShapeAttrName(state.name),
                               getAxisAttrName(state.name)))
    return mlir::failure();

  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseKeyword("row") || parseBlockArgList(parser, entryArgs))
    return mlir::failure();
  if (entryArgs.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "'map.iter.rowwise.body' expects exactly one row "
                            "argument");

  // capturing ( %inner = %outer : type, ... ), as in map.iter.elementwise.body.
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (parser.parseKeyword("capturing") || parser.parseLParen())
    return mlir::failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument innerArg;
      OpAsmParser::UnresolvedOperand outer;
      Type type;
      if (parser.parseArgument(innerArg) || parser.parseEqual() ||
          parser.parseOperand(outer) || parser.parseColonType(type))
        return mlir::failure();
      innerArg.type = type;
      entryArgs.push_back(innerArg);
      captures.push_back(outer);
      captureTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return mlir::failure();
  }
  if (parser.resolveOperands(captures, captureTypes,
                             parser.getCurrentLocation(), state.operands))
    return mlir::failure();

  Region *body = state.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterRowwiseBodyOp::print(OpAsmPrinter &p) {
  printRowwiseShapeAndAxis(p, getShape(), getAxis());

  Block &block = getBody().front();
  ValueRange captures = getCaptures();
  p.increaseIndent();
  p.printNewline();
  p << "row ";
  printBlockArgList(p, block.getArguments().take_front());
  p.printNewline();
  p << "capturing (";
  llvm::interleaveComma(
      llvm::seq<unsigned>(0, captures.size()), p, [&](unsigned i) {
        BlockArgument inner = block.getArgument(1 + i);
        p << inner << " = " << captures[i] << " : " << inner.getType();
      });
  p << ") ";
  p.decreaseIndent();

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {getShapeAttrName().getValue(), getAxisAttrName().getValue()});
}

LogicalResult IterRowwiseBodyOp::verify() {
  Block &body = getBody().front();
  unsigned numCaptures = getCaptures().size();
  if (body.getNumArguments() != 1 + numCaptures)
    return emitOpError() << "body expects exactly " << (1 + numCaptures)
                         << " block arguments (the row, then one per "
                            "capture), got "
                         << body.getNumArguments();
  return verifyRowIndex(*this, getShape(), getAxis(), body.getArgument(0));
}

//===----------------------------------------------------------------------===//
// map.iter.reduce
//===----------------------------------------------------------------------===//

static bool isScalarType(Type type) {
  return llvm::isa<FloatType, IntegerType>(type);
}

ParseResult IterReduceOp::parse(OpAsmParser &parser, OperationState &state) {
  FlatSymbolRefAttr monoid;
  if (parser.parseAttribute(monoid, getMonoidAttrName(state.name),
                            state.attributes))
    return mlir::failure();

  SmallVector<OpAsmParser::Argument> indexArgInfo;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, indexArgInfo))
    return mlir::failure();
  if (indexArgInfo.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "'map.iter.reduce' expects exactly one index "
                            "argument");

  // Optional so that a fold without results reaches the verifier.
  if (parser.parseOptionalArrowTypeList(state.types) ||
      parser.parseRegion(*state.addRegion(), indexArgInfo))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterReduceOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getMonoidAttr());
  p << " index ";
  printBlockArgList(p, getBody().front().getArguments());
  p.printArrowTypeList(getResultTypes());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {getMonoidAttrName().getValue()});
}

void IterReduceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // Printing must not fail on IR that `verifySymbolUses` would reject.
  auto decl = SymbolTable::lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getMonoidAttr());
  M::MOGG::KernelReductionAttr reduction =
      decl ? decl.getReductionAttr() : nullptr;
  if (!reduction)
    return;
  ArrayRef<Attribute> outputs = reduction.getOutputs().getValue();
  if (outputs.size() != getNumResults())
    return;
  for (auto [result, attr] : llvm::zip_equal(getResults(), outputs))
    setNameFn(result,
              llvm::cast<M::MOGG::KernelReductionOutputAttr>(attr).getName());
}

YieldOp IterReduceOp::getYield() {
  Block &body = getBody().front();
  return body.empty() ? YieldOp() : llvm::dyn_cast<YieldOp>(body.back());
}

LogicalResult IterReduceOp::verify() {
  if (getNumResults() == 0)
    return emitOpError() << "must have at least one result";
  for (auto [i, type] : llvm::enumerate(getResultTypes()))
    if (!isScalarType(type))
      return emitOpError() << "result " << i << " must be a scalar, got "
                           << type;

  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError() << "body expects exactly one block argument (the "
                            "index), got "
                         << body.getNumArguments();
  Operation *parent = (*this)->getParentOp();
  MOSH::ShapeAttr shape =
      llvm::isa<IterRowwiseOp>(parent)
          ? llvm::cast<IterRowwiseOp>(parent).getShape()
          : llvm::cast<IterRowwiseBodyOp>(parent).getShape();
  auto indexTy = llvm::dyn_cast<IndexType>(getIndexArg().getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";
  if (indexTy.getRank() != shape.getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match the rowwise shape rank "
                         << shape.getRank();

  if (!getYield())
    return emitOpError() << "body must end with a map.yield of the "
                            "contribution";
  if (getYield().getOperands().empty())
    return emitOpError() << "must yield at least one contribution";
  for (auto [i, type] : llvm::enumerate(getYield().getOperandTypes()))
    if (!isScalarType(type))
      return emitOpError() << "contribution " << i << " must be a scalar, got "
                           << type;

  WalkResult stores =
      body.walk([](StoreOp) { return WalkResult::interrupt(); });
  if (stores.wasInterrupted())
    return emitOpError() << "must not store; a fold only reads the row";
  return mlir::success();
}

LogicalResult
IterReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getMonoidAttr());
  if (!decl)
    return emitOpError("'")
           << getMonoid() << "' does not refer to a kernel.decl op";
  M::MOGG::KernelReductionAttr reduction = decl.getReductionAttr();
  if (!reduction)
    return emitOpError("kernel '") << getMonoid() << "' is not a reduction";

  ArrayRef<Attribute> outputs = reduction.getOutputs().getValue();
  if (outputs.size() != getNumResults())
    return emitOpError("monoid '") << getMonoid() << "' has " << outputs.size()
                                   << " output fields, but the fold has "
                                   << getNumResults() << " results";

  // Outputs whose dtype is the same struct parameter share one type. Fixed
  // dtypes are checked when the monoid's parameters are inferred.
  llvm::StringMap<std::pair<unsigned, Type>> paramTypes;
  for (auto [i, attr] : llvm::enumerate(outputs)) {
    auto output = llvm::cast<M::MOGG::KernelReductionOutputAttr>(attr);
    auto ref = llvm::dyn_cast<KGEN::ParamDeclRefAttr>(
        KGEN::SugarAttr::strip(output.getDtype()));
    if (!ref)
      continue;
    Type type = getResult(i).getType();
    auto [it, inserted] =
        paramTypes.try_emplace(ref.getName().strref(), i, type);
    if (!inserted && it->second.second != type)
      return emitOpError("results ")
             << it->second.first << " and " << i << " are both of dtype "
             << "parameter " << ref.getName() << " but have types "
             << it->second.second << " and " << type;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.constant.scalar
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOp::verify() {
  if (getValue().getType() != getResult().getType())
    return emitOpError("value attribute type ")
           << getValue().getType() << " does not match result type "
           << getResult().getType();
  return mlir::success();
}

OpFoldResult ConstantScalarOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// map.store
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  if (!M::MO::dynCastFromTensorLike(getTensor().getType()))
    return emitOpError()
           << "tensor operand must be !mo.tensor or !mo.buffer type";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.slice
//===----------------------------------------------------------------------===//

static LogicalResult verifySliceBoundsOperandsAreRank1(
    Operation *op, ArrayRef<std::pair<StringRef, Value>> bounds) {
  for (auto [name, value] : bounds) {
    auto tensorTy = llvm::dyn_cast<M::MO::TensorType>(value.getType());
    if (!tensorTy || tensorTy.getRank() != 1)
      return op->emitOpError()
             << name << " operand must be a rank-1 !mo.tensor";
  }
  return mlir::success();
}

LogicalResult SliceIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return verifySliceBoundsOperandsAreRank1(getOperation(),
                                           {std::pair{"starts", getStarts()},
                                            {"stops", getStops()},
                                            {"steps", getSteps()}});
}

Value SliceIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                      const IRMapping &outerValues) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return SliceStrideOp::create(builder, getLoc(), resultTy, getSpec(), tensor,
                               outerValues.lookupOrDefault(getStarts()),
                               outerValues.lookupOrDefault(getStops()),
                               outerValues.lookupOrDefault(getSteps()));
}

//===----------------------------------------------------------------------===//
// map.index.slice.select
//===----------------------------------------------------------------------===//

// The shape `index` addresses, where that is known locally: at an epilogue's
// own index, the tensor the kernel hands the epilogue; at a foreach's loop
// index (the last argument, or the first once outlined), its loop shape; at a
// select body's argument, that select's out-shape. Null elsewhere.
static MOSH::ShapeAttr getIndexedShape(Value index) {
  auto arg = llvm::dyn_cast<BlockArgument>(index);
  if (!arg)
    return {};
  Operation *owner = arg.getOwner()->getParentOp();
  if (auto select = llvm::dyn_cast<SliceSelectOp>(owner))
    return select.getSpec().getSignature().getOut();
  if (auto foreach = llvm::dyn_cast<IterElementwiseOp>(owner))
    return arg == arg.getOwner()->getArguments().back() ? foreach.getShape()
                                                        : MOSH::ShapeAttr();
  if (auto body = llvm::dyn_cast<IterElementwiseBodyOp>(owner))
    return arg.getArgNumber() == 0 ? body.getShape() : MOSH::ShapeAttr();
  auto epilogue = llvm::dyn_cast<IterEpilogueOp>(owner);
  if (!epilogue || arg.getArgNumber() != 1)
    return {};
  auto handedTy =
      llvm::dyn_cast<M::MO::TensorType>(epilogue.getResult().getType());
  return handedTy ? handedTy.getShape() : MOSH::ShapeAttr();
}

LogicalResult
SliceSelectOp::verifySpecAt(SliceSpecAttr spec, Value index,
                            function_ref<InFlightDiagnostic()> emitError) {
  auto reject = [&](const auto &...message) {
    if (emitError)
      (emitError() << ... << message);
    return failure();
  };
  MAP::ViewSignatureAttr signature = spec.getSignature();
  unsigned rank = signature.getIn().getRank();
  if (llvm::cast<IndexType>(index.getType()).getRank() != rank)
    return reject("index rank does not match spec's rank ", rank);
  if (rank == 0)
    return reject("cannot select from a slice of a rank-0 tensor");

  // There are no bounds operands to read a runtime start or step from.
  ArrayRef<int64_t> starts = spec.getStarts().asArrayRef();
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  for (auto [d, start, step] : llvm::enumerate(starts, steps)) {
    if (ShapedType::isDynamic(start) || ShapedType::isDynamic(step))
      return reject("dim ", d, ": start and step must be static");
    if (step <= 0)
      return reject("dim ", d, ": step must be positive, got ", step);
  }
  // The body runs once per stored vector, which spans the innermost dim, so
  // selecting within that dim would split the vector.
  unsigned inner = rank - 1;
  if (starts[inner] != 0 || steps[inner] != 1 ||
      signature.getIn().getValues()[inner] !=
          signature.getOut().getValues()[inner])
    return reject("the innermost dim must not be sliced");

  // The body runs at the stored vector's first index, so the index must run
  // along the in-shape's innermost dim with the vector. A transformed index,
  // such as a reshape's, can carry one vector across rows of the in-shape.
  MOSH::ShapeAttr indexed = getIndexedShape(index);
  if (!indexed)
    return reject("index must be its kernel's loop index or an enclosing "
                  "select's argument");
  if (indexed != signature.getIn())
    return reject("spec's in-shape ", signature.getIn(),
                  " does not match the shape ", indexed,
                  " its index addresses");
  return success();
}

// The body's one argument is the output index, of the input index's type, so
// it is named after an arrow instead of in a `^bb0` header.
ParseResult SliceSelectOp::parse(OpAsmParser &parser, OperationState &result) {
  SliceSpecAttr spec;
  OpAsmParser::UnresolvedOperand index;
  OpAsmParser::Argument outIndex;
  Type indexType;
  if (parser.parseLess() || parser.parseAttribute(spec) ||
      parser.parseGreater() || parser.parseLParen() ||
      parser.parseOperand(index) || parser.parseRParen() ||
      parser.parseArrow() || parser.parseLParen() ||
      parser.parseArgument(outIndex) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(indexType) ||
      parser.resolveOperand(index, indexType, result.operands))
    return failure();
  result.addAttribute(getSpecAttrName(result.name), spec);
  outIndex.type = indexType;
  return parser.parseRegion(*result.addRegion(), outIndex);
}

void SliceSelectOp::print(OpAsmPrinter &p) {
  Block &body = getBody().front();
  p << "<" << getSpec() << "> (" << getIndex() << ") -> (";
  if (body.getNumArguments() == 1)
    p << body.getArgument(0);
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), {getSpecAttrName()});
  p << " : " << getIndex().getType() << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

SmallVector<MOSH::ShapeAttr> SliceSelectOp::getDynamicShapes() {
  MAP::ViewSignatureAttr sig = getSpec().getSignature();
  if (sig.getIn().isStatic() && sig.getOut().isStatic())
    return {};
  return {sig.getIn(), sig.getOut()};
}

LogicalResult SliceSelectOp::verify() {
  if (failed(
          verifySpecAt(getSpec(), getIndex(), [&] { return emitOpError(); })))
    return failure();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  BlockArgument outIdx = getBody().front().getNumArguments() == 1
                             ? getBody().front().getArgument(0)
                             : BlockArgument();
  if (!outIdx || outIdx.getType() != getIndex().getType())
    return emitOpError() << "body must take one index of the spec's rank";

  // A load or store at the body's argument addresses the slice's output.
  // Nested ops are not verified yet, hence the checked cast.
  for (Operation *user : outIdx.getUsers()) {
    Value tensor;
    if (auto load = llvm::dyn_cast<LoadOp>(user))
      tensor = load.getTensor();
    else if (auto store = llvm::dyn_cast<StoreOp>(user))
      tensor = store.getTensor();
    M::MO::TensorType tensorTy =
        tensor ? M::MO::dynCastFromTensorLike(tensor.getType())
               : M::MO::TensorType();
    if (tensorTy && tensorTy.getShape() != signature.getOut())
      return emitOpError() << "tensor shape " << tensorTy.getShape()
                           << " at the body's index does not match the "
                              "slice's out-shape "
                           << signature.getOut();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.slice.stride
//===----------------------------------------------------------------------===//

LogicalResult SliceStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return verifySliceBoundsOperandsAreRank1(getOperation(),
                                           {std::pair{"starts", getStarts()},
                                            {"stops", getStops()},
                                            {"steps", getSteps()}});
}

//===----------------------------------------------------------------------===//
// map.index.broadcast
//===----------------------------------------------------------------------===//

LogicalResult BroadcastIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return mlir::success();
}

Value BroadcastIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                          const IRMapping &) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return BroadcastStrideOp::create(builder, getLoc(), resultTy, getSpec(),
                                   tensor);
}

//===----------------------------------------------------------------------===//
// map.index.broadcast.stride
//===----------------------------------------------------------------------===//

LogicalResult BroadcastStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.reshape
//===----------------------------------------------------------------------===//

SmallVector<MOSH::ShapeAttr> ReshapeIndexOp::getDynamicShapes() {
  MAP::ViewSignatureAttr sig = getSpec().getSignature();
  if (sig.getIn().isStatic() && sig.getOut().isStatic())
    return {};
  // The lowered op names its index's space first: the out-shape.
  return {sig.getOut(), sig.getIn()};
}

LogicalResult ReshapeIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return mlir::success();
}

Value ReshapeIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                        const IRMapping &) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return ReshapeStrideOp::create(builder, getLoc(), resultTy, getSpec(),
                                 tensor);
}

//===----------------------------------------------------------------------===//
// map.index.reshape.stride
//===----------------------------------------------------------------------===//

LogicalResult ReshapeStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.transpose
//===----------------------------------------------------------------------===//

LogicalResult TransposeIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return verifySliceBoundsOperandsAreRank1(
      getOperation(), {std::pair{"permutations", getPermutations()}});
}

Value TransposeIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                          const IRMapping &outerValues) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return TransposeStrideOp::create(
      builder, getLoc(), resultTy, getSpec(), tensor,
      outerValues.lookupOrDefault(getPermutations()));
}

//===----------------------------------------------------------------------===//
// map.index.transpose.stride
//===----------------------------------------------------------------------===//

LogicalResult TransposeStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return verifySliceBoundsOperandsAreRank1(
      getOperation(), {std::pair{"permutations", getPermutations()}});
}

//===----------------------------------------------------------------------===//
// map.constant
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }

//===----------------------------------------------------------------------===//
// map.store queries
//===----------------------------------------------------------------------===//

StoreOp M::MAP::getUniqueStore(Block &body) {
  SmallVector<StoreOp> stores = getOutputStores(body);
  ASSERT_STREAM(stores.size() == 1,
                << "expected exactly one map.store in the body");
  // A store nested in a region runs only for some elements.
  ASSERT_STREAM(stores.front()->getBlock() == &body,
                << "expected the map.store directly in the body");
  return stores.front();
}

SmallVector<StoreOp> M::MAP::getOutputStores(Block &body) {
  SmallVector<StoreOp> stores;
  for (Operation &op : body)
    op.walk<WalkOrder::PreOrder>(
        [&](StoreOp store) { stores.push_back(store); });
  ASSERT_STREAM(!stores.empty(), << "expected a map.store in the body");
  return stores;
}
