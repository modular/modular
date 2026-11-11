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

using namespace M;
using namespace mlir;
using namespace M::MAP;
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

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

//===----------------------------------------------------------------------===//
// map.iter.elementwise.body
//===----------------------------------------------------------------------===//

// Parses an optional `max_simd_width(<limit>)`.
static ParseResult parseOptionalMaxSimdWidth(OpAsmParser &parser,
                                             OperationState &state,
                                             StringAttr name) {
  if (failed(parser.parseOptionalKeyword("max_simd_width")))
    return mlir::success();
  TypedAttr limit;
  if (parser.parseLParen() || MAP::parseWidthLimit(parser, limit) ||
      parser.parseRParen())
    return mlir::failure();
  state.addAttribute(name, limit);
  return mlir::success();
}

static void printOptionalMaxSimdWidth(OpAsmPrinter &p, TypedAttr limit) {
  if (!limit)
    return;
  p << " max_simd_width(";
  MAP::printWidthLimit(p, limit);
  p << ")";
}

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
  if (parseOptionalMaxSimdWidth(parser, state,
                                getMaxSimdWidthAttrName(state.name)))
    return mlir::failure();

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
  p << "]>";
  printOptionalMaxSimdWidth(p, getMaxSimdWidthAttr());
  p << " ";

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
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {"shape", getMaxSimdWidthAttrName().getValue()});
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

//===----------------------------------------------------------------------===//
// map.iter.cache
//===----------------------------------------------------------------------===//

ParseResult IterCacheOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::Argument> indexArgInfo;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, indexArgInfo))
    return mlir::failure();
  if (indexArgInfo.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "'map.iter.cache' expects exactly one index "
                            "argument");
  if (parser.parseArrowTypeList(state.types) ||
      parser.parseRegion(*state.addRegion(), indexArgInfo))
    return mlir::failure();
  return parser.parseOptionalAttrDict(state.attributes);
}

void IterCacheOp::print(OpAsmPrinter &p) {
  p << " index ";
  printBlockArgList(p, getBody().front().getArguments());
  p.printArrowTypeList(getOperation()->getResultTypes());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

//===----------------------------------------------------------------------===//
// map.iter.map
//===----------------------------------------------------------------------===//

ParseResult IterMapOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::Argument> indexArgInfo;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, indexArgInfo))
    return mlir::failure();
  if (indexArgInfo.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "'map.iter.map' expects exactly one index "
                            "argument");
  if (parser.parseRegion(*state.addRegion(), indexArgInfo))
    return mlir::failure();
  return parser.parseOptionalAttrDict(state.attributes);
}

void IterMapOp::print(OpAsmPrinter &p) {
  p << " index ";
  printBlockArgList(p, getBody().front().getArguments());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs());
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
