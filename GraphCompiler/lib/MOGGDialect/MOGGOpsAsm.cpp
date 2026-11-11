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
// This file implements the assembly format of the MOGG dialect operations.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include <cstdlib>
#include <future>
#include <vector>

using namespace M;
using namespace MOGG;

static ParseResult parseDim(OpAsmParser &p, StringAttr &name) {
  return failure(p.parseLess() || KGEN::parseParamName(p, name) ||
                 p.parseGreater());
}

// `except(%err) {...}`: the error is the except region's only argument, and
// its type is always `!mogg.error`.
static ParseResult parseExceptRegion(OpAsmParser &parser, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseKeyword("except") ||
      parser.parseArgumentList(args, OpAsmParser::Delimiter::OptionalParen))
    return failure();
  for (OpAsmParser::Argument &arg : args)
    arg.type = ErrorType::get(parser.getContext());
  return parser.parseRegion(region, args);
}

static void printExceptRegion(OpAsmPrinter &p, Operation *, Region &region) {
  p << "except";
  if (!region.empty() && region.getNumArguments() != 0) {
    p << "(";
    llvm::interleaveComma(region.getArguments(), p);
    p << ")";
  }
  p << " ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

void TryOp::getAsmBlockArgumentNames(Region &region,
                                     OpAsmSetValueNameFn setNameFn) {
  if (&region == &getExceptRegion() && !region.empty())
    for (BlockArgument arg : region.getArguments())
      setNameFn(arg, "err");
}

static void printDim(OpAsmPrinter &p, Operation *op, StringAttr dim) {
  p << '<';
  KGEN::printParamName(p, dim);
  p << '>';
}

//===----------------------------------------------------------------------===//
// Parameter printing/parsing support in form: <P1, P2, P3>
//===----------------------------------------------------------------------===//

/// Parse an optional symbol visibility keyword, mirroring
/// `function_interface_impl::parseFunctionOp`.
static ParseResult parseSymbolVisibility(OpAsmParser &p,
                                         StringAttr &visibility) {
  StringRef keyword;
  if (succeeded(
          p.parseOptionalKeyword(&keyword, {"public", "private", "nested"})))
    visibility = p.getBuilder().getStringAttr(keyword);
  return success();
}

/// Prints the symbol visibility keyword, if present.
static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                  StringAttr visibility) {
  if (visibility)
    p << visibility.getValue() << ' ';
}

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  auto defaultType = MOSH::ShapeAttr::getDimType(p.getContext());

  auto declParser = [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    StringAttr name;
    Type type;
    if (KGEN::parseParamName(p, name) ||
        KGEN::parseColonTypeOrDefault(p, type, defaultType))
      return failure();
    decls.push_back(ParamDeclAttr::get(name, type));
    return success();
  };

  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (KGEN::parseParamDecls(p, paramDecls, declParser) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  auto defaultType = MOSH::ShapeAttr::getDimType(paramDecls.getContext());

  auto printElt = [&](ParamDeclAttr decl) {
    KGEN::printParamName(p, decl.getName());
    KGEN::printColonTypeOrDefault(p, decl.getType(), defaultType);
  };

  if (!paramDecls.empty()) {
    p << "<";
    printParamDecls(p, paramDecls, printElt);
    p << '>';
  }
}

/// `custom<OutputParamDecls>`; named apart from `parseParamDecls` because
/// `M::MO` declares a same-signature `printParamDecls` that would make the
/// generated call ambiguous.
static ParseResult parseOutputParamDecls(OpAsmParser &p,
                                         KGEN::ParamDeclArrayAttr &paramDecls) {
  return parseParamDecls(p, paramDecls);
}

static void printOutputParamDecls(OpAsmPrinter &p, Operation *op,
                                  KGEN::ParamDeclArrayAttr paramDecls) {
  printParamDecls(p, op, paramDecls);
}

static ParseResult
parseRegionArguments(OpAsmParser &p, SmallVectorImpl<Type> &resTypes,
                     SmallVectorImpl<OpAsmParser::Argument> &args) {
  auto parseArg = [&] {
    return p.parseArgument(args.emplace_back(), /*allowType=*/true);
  };
  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseArg) ||
      p.parseOptionalArrowTypeList(resTypes))
    return failure();
  return success();
}

static void printRegionArguments(OpAsmPrinter &p, TypeRange resTypes,
                                 ArrayRef<BlockArgument> args) {
  p << '(';
  llvm::interleaveComma(args, p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ')';
  if (resTypes.empty())
    return;
  p << ' ';
  p.printArrowTypeList(resTypes);
}

//===----------------------------------------------------------------------===//
// RoutineOp
//===----------------------------------------------------------------------===//

ParseResult RoutineOp::parse(OpAsmParser &parser, OperationState &result) {
  DispatchKindAttr dispatch;
  if (parseOptionalDispatch(parser, dispatch))
    return failure();
  if (dispatch)
    result.addAttribute(getDispatchAttrName(result.name), dispatch);

  // Op operands: the `!mo.tensor` inputs, printed bare as `(%a, %b : types)`
  // (no `inputs` keyword).
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  SmallVector<Type> inputTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(inputOperands) || parser.parseColon() ||
        parser.parseTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }
  if (parser.resolveOperands(inputOperands, inputTypes, operandsLoc,
                             result.operands))
    return failure();
  int32_t numInputOperands = static_cast<int32_t>(result.operands.size());

  // Optional chain (%c) clause -- a plain operand, never a body block
  // argument (mirrors `map.iter.elementwise`'s `chain(...)`).
  int32_t numChainOperands = 0;
  if (succeeded(parser.parseOptionalKeyword("chain"))) {
    OpAsmParser::UnresolvedOperand chainOperand;
    if (parser.parseLParen() || parser.parseOperand(chainOperand) ||
        parser.parseRParen())
      return failure();
    if (parser.resolveOperand(chainOperand,
                              MO::ChainType::get(parser.getContext()),
                              result.operands))
      return failure();
    numChainOperands = 1;
  }
  result.addAttribute(RoutineOp::getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {numInputOperands, numChainOperands}));

  // Results: an optional `-> types`.
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  // Body entry-block arguments given as labeled groups. The display order is
  // inputs, outputs, context, dims; the in-memory order the verifier and
  // downstream passes rely on is [inputs, outputs, dims, context].
  // An argument prints its type only when it differs from the group's implied
  // type: tensor handles are `!mogg.pointer`, contexts `!mogg.context`.
  SmallVector<OpAsmParser::Argument> inputArgs, outputArgs, ctxArgs, dimArgs;
  auto parseGroup = [&](StringRef keyword,
                        SmallVectorImpl<OpAsmParser::Argument> &into,
                        Type impliedType) -> ParseResult {
    if (failed(parser.parseOptionalKeyword(keyword)))
      return success();
    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
          OpAsmParser::Argument &arg = into.emplace_back();
          if (parser.parseArgument(arg))
            return failure();
          arg.type = impliedType;
          if (succeeded(parser.parseOptionalColon()))
            return parser.parseType(arg.type);
          if (!arg.type)
            return parser.emitError(parser.getCurrentLocation(),
                                    "expected ':' and a type");
          return success();
        });
  };
  MLIRContext *ctx = parser.getContext();
  if (parseGroup("inputs", inputArgs, PointerType::get(ctx)) ||
      parseGroup("outputs", outputArgs, PointerType::get(ctx)) ||
      parseGroup("context", ctxArgs, ContextType::get(ctx)) ||
      parseGroup("dims", dimArgs, /*impliedType=*/nullptr))
    return failure();

  SmallVector<OpAsmParser::Argument> bodyArgs;
  bodyArgs.append(inputArgs.begin(), inputArgs.end());
  bodyArgs.append(outputArgs.begin(), outputArgs.end());
  bodyArgs.append(dimArgs.begin(), dimArgs.end());
  bodyArgs.append(ctxArgs.begin(), ctxArgs.end());

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return parser.parseRegion(*result.addRegion(), bodyArgs);
}

void RoutineOp::print(OpAsmPrinter &p) {
  if (getDispatch() != DispatchKind::Deferred) {
    p << " ";
    printOptionalDispatch(p, getDispatchAttr(), DispatchKind::Deferred);
  }
  // Op operands: the `!mo.tensor` inputs, printed bare.
  p << " (";
  p.printOperands(getInputs());
  if (!getInputs().empty()) {
    p << " : ";
    llvm::interleaveComma(getInputs().getTypes(), p);
  }
  p << ")";

  // chain (%c) -- optional plain operand, no block arg (mirrors
  // `map.iter.elementwise`'s `chain(...)`).
  if (Value chain = getChain())
    p << " chain(" << chain << ")";

  // Results: `-> types` (omitted when there are none).
  if (!getOutputs().empty()) {
    SmallVector<Type> outTypes(getOutputs().getTypes());
    p.printArrowTypeList(outTypes);
  }

  // Body entry-block arguments, as labeled groups. The in-memory order is
  // [inputs, outputs, dims, context]; the display shows context before dims.
  // Only destination-passing outputs get a body block argument: not the
  // trailing chain, and not a result that leaves by value.
  auto args = getBody().front().getArguments();
  size_t numInputs = getInputs().size();
  size_t numOutputs = 0;
  for (Value output : getOutputs())
    if (MO::dynCastFromTensorLike(output.getType()))
      ++numOutputs;
  size_t numTensors = std::min(numInputs + numOutputs, args.size());

  auto inputArgs = args.take_front(std::min(numInputs, args.size()));
  auto outputArgs = args.drop_front(inputArgs.size()).take_front(numOutputs);
  auto tail = args.drop_front(numTensors);
  // `tail` is `[dims..., context]`; the context is the trailing argument.
  BlockArgument ctxArg = tail.empty() ? BlockArgument() : tail.back();
  auto dimArgs = tail.empty() ? tail : tail.drop_back(1);

  p.increaseIndent();
  auto printGroup = [&](StringRef label, auto group, Type impliedType) {
    if (group.empty())
      return;
    p.printNewline();
    p << label << " (";
    llvm::interleaveComma(group, p, [&](BlockArgument ba) {
      if (ba.getType() == impliedType)
        p.printOperand(ba);
      else
        p.printRegionArgument(ba);
    });
    p << ")";
  };
  MLIRContext *ctx = getContext();
  printGroup("inputs", inputArgs, PointerType::get(ctx));
  printGroup("outputs", outputArgs, PointerType::get(ctx));
  if (ctxArg)
    printGroup("context", ArrayRef<BlockArgument>(ctxArg),
               ContextType::get(ctx));
  printGroup("dims", dimArgs, /*impliedType=*/Type());
  p.decreaseIndent();

  // Elided when empty.
  SmallVector<StringRef> elided{getOperandSegmentSizesAttrName().getValue()};
  if (getOutputParamDecls().empty())
    elided.push_back(getOutputParamDeclsAttrName().getValue());
  elided.push_back(getDispatchAttrName().getValue());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// Fused tensor bodies: tensor.create.epilogue, tensor.create.prologue and
// call.foreach
//
// Each region's arguments are leading values the kernel hands the body (the
// value and the index), then one per capture, then one per shape capture. The
// syntax binds every capture to its block argument by name:
//
//   %shape (%val : f32, %idx : !map.index<2>)
//       capturing {%cap = %t : !mogg.tensor} shapes {%shape0 = %s} ...
//===----------------------------------------------------------------------===//

/// Parses the bare case name an op format binds `io` with, as in
/// `io = Input`. Outside an op format the attribute prints as `#mo.io<Input>`.
static ParseResult parseIOKind(OpAsmParser &parser, MO::IOKindAttr &io) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<MO::IOKind> kind = MO::symbolizeIOKind(keyword);
  if (!kind)
    return parser.emitError(loc) << "unknown io kind '" << keyword << "'";
  io = MO::IOKindAttr::get(parser.getContext(), *kind);
  return success();
}

static void printIOKind(OpAsmPrinter &p, Operation *, MO::IOKindAttr io) {
  p << MO::stringifyIOKind(io.getValue());
}

/// Parses the attributes the fused tensor creates bind, as their declarative
/// siblings print them: `[computeOutputArg = true] [contract = <...>]
/// device = "...", elementType = T, io = K, mut = B`. `computeOutputArg` is
/// only accepted when `computeOutputArgName` is set.
static ParseResult
parseFusedTensorAttrs(OpAsmParser &parser, OperationState &result,
                      StringAttr computeOutputArgName, StringAttr contractName,
                      StringAttr deviceName, StringAttr elementTypeName,
                      StringAttr ioName, StringAttr mutName) {
  if (computeOutputArgName &&
      succeeded(parser.parseOptionalKeyword("computeOutputArg"))) {
    BoolAttr computeOutputArg;
    if (parser.parseEqual() || parser.parseAttribute(computeOutputArg))
      return failure();
    result.addAttribute(computeOutputArgName, computeOutputArg);
  }
  if (contractName && succeeded(parser.parseOptionalKeyword("contract"))) {
    TensorArgContractAttr contract;
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(contract))
      return failure();
    result.addAttribute(contractName, contract);
  }
  StringAttr device;
  Type elementType;
  MO::IOKindAttr io;
  BoolAttr mut;
  if (parser.parseKeyword("device") || parser.parseEqual() ||
      parser.parseAttribute(device) || parser.parseComma() ||
      parser.parseKeyword("elementType") || parser.parseEqual() ||
      parser.parseType(elementType) || parser.parseComma() ||
      parser.parseKeyword("io") || parser.parseEqual() ||
      parseIOKind(parser, io) || parser.parseComma() ||
      parser.parseKeyword("mut") || parser.parseEqual() ||
      parser.parseAttribute(mut))
    return failure();
  result.addAttribute(deviceName, device);
  result.addAttribute(elementTypeName, TypeAttr::get(elementType));
  result.addAttribute(ioName, io);
  result.addAttribute(mutName, mut);
  return success();
}

static void printFusedTensorAttrs(OpAsmPrinter &p, bool computeOutputArg,
                                  TensorArgContractAttr contract,
                                  StringAttr device, Type elementType,
                                  MO::IOKindAttr io, bool mut) {
  if (computeOutputArg)
    p << " computeOutputArg = true";
  if (contract) {
    p << " contract = ";
    p.printStrippedAttrOrType(contract);
  }
  p << " device = ";
  p.printAttribute(device);
  p << ", elementType = " << elementType << ", io = ";
  printIOKind(p, nullptr, io);
  p << ", mut = " << (mut ? "true" : "false");
}

static void nameArgs(ArrayRef<BlockArgument> args, StringRef prefix,
                     OpAsmSetValueNameFn setNameFn) {
  for (auto [i, arg] : llvm::enumerate(args))
    setNameFn(arg, args.size() == 1 ? prefix.str() : (prefix + Twine(i)).str());
}

static void nameLeadingArgs(ArrayRef<BlockArgument> args,
                            ArrayRef<StringRef> names,
                            OpAsmSetValueNameFn setNameFn) {
  if (args.size() != names.size())
    return;
  for (auto [arg, name] : llvm::zip_equal(args, names))
    setNameFn(arg, name);
}

/// Parses `(%leading...) capturing {...} shapes {...}` into `entryArgs`,
/// ordered as the region expects them.
static ParseResult
parseFusedBodyArgs(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captures,
                   SmallVectorImpl<Type> &captureTypes,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapes) {
  SmallVector<OpAsmParser::Argument> shapeArgs;
  SmallVector<Type> shapeTypes;
  if (parseLeadingArgs(parser, entryArgs) ||
      parseBindings(parser, "capturing", entryArgs, captures, captureTypes,
                    /*impliedType=*/Type(), /*required=*/false) ||
      parseBindings(parser, "shapes", shapeArgs, shapes, shapeTypes,
                    ShapeType::get(parser.getContext()), /*required=*/false))
    return failure();
  entryArgs.append(shapeArgs.begin(), shapeArgs.end());
  return success();
}

static void printFusedBodyArgs(OpAsmPrinter &p, Block &body,
                               ValueRange captures, ValueRange shapes) {
  auto args = body.getArguments();
  size_t numLeading = args.size() - captures.size() - shapes.size();
  p << " ";
  printLeadingArgs(p, args.take_front(numLeading));
  if (!captures.empty()) {
    p << " ";
    printBindings(p, "capturing", captures,
                  args.slice(numLeading, captures.size()),
                  /*impliedType=*/Type(), /*required=*/false);
  }
  if (!shapes.empty()) {
    p << " ";
    printBindings(p, "shapes", shapes, args.take_back(shapes.size()),
                  ShapeType::get(body.getParent()->getContext()),
                  /*required=*/false);
  }
}

static void nameFusedBodyArgs(Block &body, size_t numCaptures, size_t numShapes,
                              ArrayRef<StringRef> leading,
                              OpAsmSetValueNameFn setNameFn) {
  auto args = body.getArguments();
  size_t numLeading = args.size() - numCaptures - numShapes;
  nameLeadingArgs(args.take_front(numLeading), leading, setNameFn);
  nameArgs(args.slice(numLeading, numCaptures), "cap", setNameFn);
  nameArgs(args.take_back(numShapes), "shape", setNameFn);
}

ParseResult TensorCreateEpilogueOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  OpAsmParser::UnresolvedOperand shape, outputPtr;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captures, shapes;
  SmallVector<Type> captureTypes;
  if (parser.parseOperand(shape) ||
      parseFusedBodyArgs(parser, entryArgs, captures, captureTypes, shapes) ||
      parser.parseKeyword("ptr") || parser.parseOperand(outputPtr) ||
      parseFusedTensorAttrs(
          parser, result, getComputeOutputArgAttrName(result.name),
          getContractAttrName(result.name), getDeviceAttrName(result.name),
          getElementTypeAttrName(result.name), getIoAttrName(result.name),
          getMutAttrName(result.name)) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Type shapeTy = ShapeType::get(ctx);
  if (parser.resolveOperand(shape, shapeTy, result.operands) ||
      parser.resolveOperands(captures, captureTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(shapes, shapeTy, result.operands) ||
      parser.resolveOperand(outputPtr, PointerType::get(ctx), result.operands))
    return failure();
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(captures.size()),
                           static_cast<int32_t>(shapes.size()), 1}));
  result.addTypes(TensorType::get(ctx));
  return parser.parseRegion(*result.addRegion(), entryArgs);
}

void TensorCreateEpilogueOp::print(OpAsmPrinter &p) {
  p << " " << getShape();
  printFusedBodyArgs(p, getBody().front(), getCaptures(), getShapeCaptures());
  p << " ptr " << getOutputPtr();
  printFusedTensorAttrs(p, getComputeOutputArg(), getContractAttr(),
                        getDeviceAttr(), getElementType(), getIoAttr(),
                        getMut());
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getOperandSegmentSizesAttrName().getValue(),
       getComputeOutputArgAttrName().getValue(),
       getContractAttrName().getValue(), getDeviceAttrName().getValue(),
       getElementTypeAttrName().getValue(), getIoAttrName().getValue(),
       getMutAttrName().getValue()});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void TensorCreateEpilogueOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  if (!region.empty())
    nameFusedBodyArgs(region.front(), getCaptures().size(),
                      getShapeCaptures().size(), {"val", "idx"}, setNameFn);
}

ParseResult TileCreateEpilogueOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  OpAsmParser::UnresolvedOperand shape, outputPtr;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captures, shapes;
  SmallVector<Type> captureTypes;
  TileFusionLaneAttr lane;
  llvm::SMLoc argsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(shape) ||
      parseFusedBodyArgs(parser, entryArgs, captures, captureTypes, shapes))
    return failure();
  // A tile body moves whole tiles at the kernel's index, so it has no
  // index-transforming op to thread runtime shapes into.
  if (!shapes.empty())
    return parser.emitError(argsLoc, "a tile epilogue takes no shape captures");
  if (parser.parseKeyword("ptr") || parser.parseOperand(outputPtr) ||
      parseTileFusionLane(parser, lane) || parser.parseComma() ||
      parseFusedTensorAttrs(
          parser, result, /*computeOutputArgName=*/{}, /*contractName=*/{},
          getDeviceAttrName(result.name), getElementTypeAttrName(result.name),
          getIoAttrName(result.name), getMutAttrName(result.name)) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  result.addAttribute(getLaneAttrName(result.name), lane);

  if (parser.resolveOperand(shape, ShapeType::get(ctx), result.operands) ||
      parser.resolveOperands(captures, captureTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperand(outputPtr, PointerType::get(ctx), result.operands))
    return failure();
  result.addTypes(TensorType::get(ctx));
  return parser.parseRegion(*result.addRegion(), entryArgs);
}

void TileCreateEpilogueOp::print(OpAsmPrinter &p) {
  p << " " << getShape();
  printFusedBodyArgs(p, getBody().front(), getCaptures(), /*shapes=*/{});
  p << " ptr " << getOutputPtr() << " ";
  printTileFusionLane(p, getLaneAttr());
  p << ",";
  printFusedTensorAttrs(p, /*computeOutputArg=*/false, /*contract=*/{},
                        getDeviceAttr(), getElementType(), getIoAttr(),
                        getMut());
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getLaneAttrName().getValue(), getDeviceAttrName().getValue(),
       getElementTypeAttrName().getValue(), getIoAttrName().getValue(),
       getMutAttrName().getValue()});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void TileCreateEpilogueOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  if (!region.empty())
    nameFusedBodyArgs(region.front(), getCaptures().size(), /*numShapes=*/0,
                      {"val", "idx"}, setNameFn);
}

ParseResult TensorCreatePrologueOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  OpAsmParser::UnresolvedOperand shape;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captures, shapes;
  SmallVector<Type> captureTypes;
  if (parser.parseOperand(shape) ||
      parseFusedBodyArgs(parser, entryArgs, captures, captureTypes, shapes) ||
      parseFusedTensorAttrs(
          parser, result, /*computeOutputArgName=*/{},
          getContractAttrName(result.name), getDeviceAttrName(result.name),
          getElementTypeAttrName(result.name), getIoAttrName(result.name),
          getMutAttrName(result.name)) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Type shapeTy = ShapeType::get(ctx);
  if (parser.resolveOperand(shape, shapeTy, result.operands) ||
      parser.resolveOperands(captures, captureTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(shapes, shapeTy, result.operands))
    return failure();
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(captures.size()),
                           static_cast<int32_t>(shapes.size())}));
  result.addTypes(TensorType::get(ctx));
  return parser.parseRegion(*result.addRegion(), entryArgs);
}

void TensorCreatePrologueOp::print(OpAsmPrinter &p) {
  p << " " << getShape();
  printFusedBodyArgs(p, getBody().front(), getCaptures(), getShapeCaptures());
  printFusedTensorAttrs(p, /*computeOutputArg=*/false, getContractAttr(),
                        getDeviceAttr(), getElementType(), getIoAttr(),
                        getMut());
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getOperandSegmentSizesAttrName().getValue(),
       getContractAttrName().getValue(), getDeviceAttrName().getValue(),
       getElementTypeAttrName().getValue(), getIoAttrName().getValue(),
       getMutAttrName().getValue()});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void TensorCreatePrologueOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  if (!region.empty())
    nameFusedBodyArgs(region.front(), getCaptures().size(),
                      getShapeCaptures().size(), {"idx"}, setNameFn);
}

// Unlike the epilogue and prologue, the foreach body takes its captures first
// and the index last.
ParseResult CallForeachOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  FlatSymbolRefAttr callee;
  OpAsmParser::UnresolvedOperand shape, context;
  SmallVector<OpAsmParser::Argument> leadingArgs, entryArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (parser.parseAttribute(callee, getCalleeAttrName(result.name),
                            result.attributes) ||
      parser.parseKeyword("shape") || parser.parseLParen() ||
      parser.parseOperand(shape) || parser.parseRParen() ||
      parser.parseKeyword("context") || parser.parseLParen() ||
      parser.parseOperand(context) || parser.parseRParen() ||
      parseLeadingArgs(parser, leadingArgs) ||
      parseBindings(parser, "capturing", entryArgs, captures, captureTypes,
                    /*impliedType=*/Type(), /*required=*/false))
    return failure();
  StringAttr device;
  if (parser.parseKeyword("device") || parser.parseEqual() ||
      parser.parseAttribute(device) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  result.addAttribute(getDeviceAttrName(result.name), device);
  entryArgs.append(leadingArgs.begin(), leadingArgs.end());

  if (parser.resolveOperand(shape, ShapeType::get(ctx), result.operands) ||
      parser.resolveOperand(context, ContextType::get(ctx), result.operands) ||
      parser.resolveOperands(captures, captureTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  return parser.parseRegion(*result.addRegion(), entryArgs);
}

void CallForeachOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getCalleeAttr());
  p << " shape(" << getShape() << ") context(" << getContext() << ") ";
  auto args = getBody().front().getArguments();
  size_t numCaptures = getCapturing().size();
  printLeadingArgs(p, args.drop_front(numCaptures));
  if (numCaptures) {
    p << " ";
    printBindings(p, "capturing", getCapturing(), args.take_front(numCaptures),
                  /*impliedType=*/Type(),
                  /*required=*/false);
  }
  p << " device = ";
  p.printAttribute(getDeviceAttr());
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getCalleeAttrName().getValue(), getDeviceAttrName().getValue()});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void CallForeachOp::getAsmBlockArgumentNames(Region &region,
                                             OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  size_t numCaptures = getCapturing().size();
  nameArgs(args.take_front(numCaptures), "cap", setNameFn);
  nameLeadingArgs(args.drop_front(numCaptures), {"idx"}, setNameFn);
}

// Same layout as `call.foreach`.
ParseResult CallForeachTileOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  FlatSymbolRefAttr callee;
  OpAsmParser::UnresolvedOperand shape, context;
  SmallVector<OpAsmParser::Argument> leadingArgs, entryArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  StringAttr device;
  if (parser.parseAttribute(callee, getCalleeAttrName(result.name),
                            result.attributes) ||
      parser.parseKeyword("shape") || parser.parseLParen() ||
      parser.parseOperand(shape) || parser.parseRParen() ||
      parser.parseKeyword("context") || parser.parseLParen() ||
      parser.parseOperand(context) || parser.parseRParen() ||
      parseLeadingArgs(parser, leadingArgs) ||
      parseBindings(parser, "capturing", entryArgs, captures, captureTypes,
                    /*impliedType=*/Type(), /*required=*/false) ||
      parser.parseKeyword("device") || parser.parseEqual() ||
      parser.parseAttribute(device) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  result.addAttribute(getDeviceAttrName(result.name), device);
  entryArgs.append(leadingArgs.begin(), leadingArgs.end());

  if (parser.resolveOperand(shape, ShapeType::get(ctx), result.operands) ||
      parser.resolveOperand(context, ContextType::get(ctx), result.operands) ||
      parser.resolveOperands(captures, captureTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  return parser.parseRegion(*result.addRegion(), entryArgs);
}

void CallForeachTileOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getCalleeAttr());
  p << " shape(" << getShape() << ") context(" << getContext() << ") ";
  auto args = getBody().front().getArguments();
  size_t numCaptures = getCapturing().size();
  printLeadingArgs(p, args.drop_front(numCaptures));
  if (numCaptures) {
    p << " ";
    printBindings(p, "capturing", getCapturing(), args.take_front(numCaptures),
                  /*impliedType=*/Type(),
                  /*required=*/false);
  }
  p << " device = ";
  p.printAttribute(getDeviceAttr());
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getCalleeAttrName().getValue(), getDeviceAttrName().getValue()});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void CallForeachTileOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  size_t numCaptures = getCapturing().size();
  nameArgs(args.take_front(numCaptures), "cap", setNameFn);
  nameLeadingArgs(args.drop_front(numCaptures), {"idx"}, setNameFn);
}

ParseResult CallRowwiseOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  OpAsmParser::UnresolvedOperand shape, context;
  int64_t axis;
  SmallVector<OpAsmParser::Argument> leadingArgs, entryArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (parser.parseKeyword("shape") || parser.parseLParen() ||
      parser.parseOperand(shape) || parser.parseRParen() ||
      parser.parseKeyword("axis") || parser.parseLParen() ||
      parser.parseInteger(axis) || parser.parseRParen() ||
      parser.parseKeyword("context") || parser.parseLParen() ||
      parser.parseOperand(context) || parser.parseRParen() ||
      parseLeadingArgs(parser, leadingArgs) ||
      parseBindings(parser, "capturing", entryArgs, captures, captureTypes,
                    /*impliedType=*/Type(), /*required=*/false))
    return failure();
  StringAttr device;
  if (parser.parseKeyword("device") || parser.parseEqual() ||
      parser.parseAttribute(device) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  result.addAttribute(getAxisAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(axis));
  result.addAttribute(getDeviceAttrName(result.name), device);
  entryArgs.append(leadingArgs.begin(), leadingArgs.end());

  if (parser.resolveOperand(shape, ShapeType::get(ctx), result.operands) ||
      parser.resolveOperand(context, ContextType::get(ctx), result.operands) ||
      parser.resolveOperands(captures, captureTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  return parser.parseRegion(*result.addRegion(), entryArgs);
}

void CallRowwiseOp::print(OpAsmPrinter &p) {
  p << " shape(" << getShape() << ") axis(" << getAxis() << ") context("
    << getContext() << ") ";
  auto args = getBody().front().getArguments();
  size_t numCaptures = getCapturing().size();
  printLeadingArgs(p, args.drop_front(numCaptures));
  if (numCaptures) {
    p << " ";
    printBindings(p, "capturing", getCapturing(), args.take_front(numCaptures),
                  /*impliedType=*/Type(),
                  /*required=*/false);
  }
  p << " device = ";
  p.printAttribute(getDeviceAttr());
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getAxisAttrName().getValue(), getDeviceAttrName().getValue()});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void CallRowwiseOp::getAsmBlockArgumentNames(Region &region,
                                             OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  size_t numCaptures = getCapturing().size();
  nameArgs(args.take_front(numCaptures), "cap", setNameFn);
  nameLeadingArgs(args.drop_front(numCaptures), {"row"}, setNameFn);
}

ParseResult RowwiseReduceOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  FlatSymbolRefAttr monoid;
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseAttribute(monoid, getMonoidAttrName(result.name),
                            result.attributes) ||
      parseLeadingArgs(parser, entryArgs) ||
      parser.parseOptionalArrowTypeList(result.types) ||
      parser.parseRegion(*result.addRegion(), entryArgs))
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

void RowwiseReduceOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getMonoidAttr());
  p << " ";
  printLeadingArgs(p, getBody().front().getArguments());
  p.printArrowTypeList(getResultTypes());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getMonoidAttrName().getValue()});
}

void RowwiseReduceOp::getAsmBlockArgumentNames(Region &region,
                                               OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  nameLeadingArgs(region.front().getArguments(), {"i"}, setNameFn);
}

ParseResult RowwiseCacheOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parseLeadingArgs(parser, entryArgs) ||
      parser.parseArrowTypeList(result.types) ||
      parser.parseRegion(*result.addRegion(), entryArgs))
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

void RowwiseCacheOp::print(OpAsmPrinter &p) {
  p << " ";
  printLeadingArgs(p, getBody().front().getArguments());
  p.printArrowTypeList(getOperation()->getResultTypes());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

void RowwiseCacheOp::getAsmBlockArgumentNames(Region &region,
                                              OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  nameLeadingArgs(region.front().getArguments(), {"i"}, setNameFn);
}

ParseResult RowwiseMapOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parseLeadingArgs(parser, entryArgs) ||
      parser.parseRegion(*result.addRegion(), entryArgs))
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

void RowwiseMapOp::print(OpAsmPrinter &p) {
  p << " ";
  printLeadingArgs(p, getBody().front().getArguments());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

void RowwiseMapOp::getAsmBlockArgumentNames(Region &region,
                                            OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  nameLeadingArgs(region.front().getArguments(), {"i"}, setNameFn);
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef kernelStubOpDisallowedAttrNames[] = {
    "name",          "returnType",     "sym_name",
    "function_type", "sym_visibility", "dispatch"};

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  StringAttr name;
  if (failed(parser.parseSymbolName(name)))
    return failure();
  state.addAttribute(getSymNameAttrName(state.name), name);
  // Optional: without the keyword, `sym_visibility` keeps its default.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, state.attributes);
  DispatchKindAttr dispatch;
  if (parseOptionalDispatch(parser, dispatch))
    return failure();
  if (dispatch)
    state.addAttribute(getDispatchAttrName(state.name), dispatch);

  SmallVector<Type> resTypes;
  SmallVector<OpAsmParser::Argument> args;
  if (failed(parseRegionArguments(parser, resTypes, args)))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &&arg : args)
    argTypes.push_back(arg.type);

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(FunctionType::get(parser.getContext(),
                                                     argTypes, resTypes)));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : kernelStubOpDisallowedAttrNames) {
    if (parsedAttributes.get(disallowed)) {
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
    }
  }
  state.attributes.append(parsedAttributes);

  if (failed(parser.parseRegion(*state.addRegion(), args)))
    return failure();

  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymNameAttr());
  p << " " << getSymVisibility() << " ";
  if (printOptionalDispatch(p, getDispatchAttr(), DispatchKind::Immediate))
    p << " ";
  printRegionArguments(p, getFunctionType().getResults(),
                       getBody().getArguments());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     kernelStubOpDisallowedAttrNames);
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// AsyncRoutineOp Parsing/Printing
//===----------------------------------------------------------------------===//

static ParseResult parseKernel(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes, SmallVectorImpl<Type> &resultTypes,
    Region &region, KGEN::ParamDeclArrayAttr &resultParams) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;

  llvm::SMLoc sigLoc;
  if (parseParamDecls(p, resultParams) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() || p.getCurrentLocation(&sigLoc) ||
      parseRegionArguments(p, resultTypes, args))
    return failure();

  // The signature is written in terms of the operand (element) types `T`; the
  // operands are `T` while the region body arguments are `!mogg.async<T>`.
  for (auto &arg : args)
    operandTypes.push_back(arg.type);

  // For AsyncRoutineOp, if the kernel takes in a chain, there will be
  // 1 fewer argument than operands.
  operandTypes.resize(operands.size(), MO::ChainType::get(p.getContext()));

  // Wrap the parsed (element) types into `!mogg.async<T>` for the body block
  // arguments before creating the region.
  for (auto &arg : args)
    arg.type = AsyncType::get(arg.type);

  if (p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();

  return success();
}

static void printKernel(OpAsmPrinter &p, Operation *kernel, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes,
                        Region &region, KGEN::ParamDeclArrayAttr resultParams) {
  printParamDecls(p, kernel, resultParams);
  p << '(';
  p.printOperands(operands);
  p << ") : ";

  // The body block arguments are `!mogg.async<T>`; print the unwrapped element
  // type `T` so the signature reads in terms of the operand types.
  p << '(';
  llvm::interleaveComma(region.getArguments(), p, [&](BlockArgument arg) {
    p.printOperand(arg);
    p << ": ";
    Type type = arg.getType();
    if (auto asyncType = dyn_cast<AsyncType>(type))
      type = asyncType.getElementType();
    p.printType(type);
    // `parseArgument` reads the location after the type, but
    // `printRegionArgument` with `omitType` prints it before.
    p.printOptionalLocationSpecifier(arg.getLoc());
  });
  p << ')';
  if (!resultTypes.empty()) {
    p << ' ';
    p.printArrowTypeList(resultTypes);
    p << ' ';
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

using PointerType = KGEN::PointerType;

#define GET_OP_CLASSES
#include "GraphCompiler/MOGGDialect/MOGG.cpp.inc"
