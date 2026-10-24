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
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/Permutation.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

static SmallVector<ChainValue> getChainValues(ValueRange values) {
  SmallVector<ChainValue> results;

  for (auto value : values) {
    if (auto chain = dyn_cast<ChainValue>(value))
      results.push_back(chain);
  }

  return results;
}

static ParseResult
parseParamDeclElement(OpAsmParser &parser,
                      SmallVectorImpl<ParamDeclAttr> &decls) {
  auto dimType = ShapeAttr::getDimType(parser.getContext());

  StringAttr name;
  Type type;
  if (KGEN::parseParamName(parser, name) ||
      KGEN::parseColonTypeOrDefault(parser, type, dimType))
    return failure();

  decls.push_back(ParamDeclAttr::get(name, type));
  return success();
}

static void printParamDeclElement(OpAsmPrinter &printer,
                                  KGEN::ParamDeclAttr decl) {
  auto dimType = ShapeAttr::getDimType(decl.getContext());
  KGEN::printParamName(printer, decl.getName());
  KGEN::printColonTypeOrDefault(printer, decl.getType(), dimType);
}

ParseResult parseParamDecls(OpAsmParser &p,
                            KGEN::ParamDeclArrayAttr &paramDecls) {
  auto parserHook = [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    return parseParamDeclElement(p, decls);
  };

  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (p.parseLParen() || p.parseRParen() || p.parseArrow() ||
        KGEN::parseParamDecls(p, paramDecls, parserHook) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

void printParamDecls(OpAsmPrinter &p, Operation *,
                     KGEN::ParamDeclArrayAttr paramDecls) {
  auto printerHook = [&](KGEN::ParamDeclAttr decl) {
    printParamDeclElement(p, decl);
  };

  if (!paramDecls.empty()) {
    p << "<() -> ";
    KGEN::printParamDecls(p, paramDecls, printerHook);
    p << '>';
  }
}

/// Parse the result tensor device in MO::TransferOp.
ParseResult parseTransferResult(AsmParser &p, Type inputType,
                                Type &resultType) {
  DeviceRefAttr deviceRef;
  Attribute parsedDevice = DeviceRefAttr::parse(p, /*odsType=*/{});
  if (!parsedDevice)
    return failure();
  deviceRef = cast<DeviceRefAttr>(parsedDevice);
  resultType = cast<TensorType>(inputType).cloneWith(deviceRef);
  return success();
}

/// Print the result tensor device in MO::TransferOp.
void printTransferResult(AsmPrinter &p, Operation *op, TensorType inputType,
                         TensorType resultType) {
  DeviceRefAttr deviceRef = resultType.getDeviceRef();
  ASSERT_STREAM(deviceRef, "DeviceRefAttr should be present.");
  deviceRef.print(p);
}

void printAttrGroup(AsmPrinter &p, ArrayRef<AttrGroupElement> elements) {
  SmallVector<const AttrGroupElement *> present;
  for (const AttrGroupElement &element : elements)
    if (element.value)
      present.push_back(&element);
  if (present.empty())
    return;

  p << " [";
  llvm::interleaveComma(present, p, [&](const AttrGroupElement *element) {
    p << element->name << " = ";
    // String/bool/integer attributes are self-describing without their
    // type, but an elements attribute is not (`#M.dense_array<1.5>` needs
    // its `: tensor<f32>` to re-parse), so print those in full.
    if (isa<ElementsAttr>(element->value))
      p << element->value;
    else
      p.printAttributeWithoutType(element->value);
  });
  p << ']';
}

ParseResult parseAttrGroup(AsmParser &p, ArrayRef<AttrGroupParser> elements) {
  SmallVector<StringRef> seen;
  llvm::SMLoc groupLoc = p.getCurrentLocation();
  auto parseElement = [&]() -> ParseResult {
    llvm::SMLoc loc = p.getCurrentLocation();
    StringRef name;
    if (p.parseKeyword(&name))
      return failure();
    const auto *element = llvm::find_if(
        elements, [&](const AttrGroupParser &e) { return e.name == name; });
    if (element == elements.end())
      return p.emitError(loc) << "unexpected attribute name '" << name << "'";
    if (llvm::is_contained(seen, name))
      return p.emitError(loc) << "duplicate attribute name '" << name << "'";
    seen.push_back(name);
    return failure(p.parseEqual() || element->parse());
  };

  if (succeeded(p.parseOptionalLSquare())) {
    do {
      if (parseElement())
        return failure();
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRSquare())
      return failure();
  }

  for (const AttrGroupParser &element : elements)
    if (element.required && !llvm::is_contained(seen, element.name))
      return p.emitError(groupLoc) << "expected '" << element.name << "'";
  return success();
}

/// `[name = value]` element for a default-valued attribute, dropped when the
/// attribute still holds its default.
static AttrGroupElement nonDefault(StringRef name, BoolAttr attr, bool dflt) {
  bool set = attr && attr.getValue() != dflt;
  return {name, set ? Attribute(attr) : Attribute()};
}

static AttrGroupElement nonDefault(StringRef name, IntegerAttr attr,
                                   int64_t dflt) {
  bool set = attr && attr.getInt() != dflt;
  return {name, set ? Attribute(attr) : Attribute()};
}

static AttrGroupElement nonDefault(StringRef name, StringAttr attr,
                                   StringRef dflt) {
  bool set = attr && attr.getValue() != dflt;
  return {name, set ? Attribute(attr) : Attribute()};
}

ParseResult parseMatmulAttrs(OpAsmParser &p, BoolAttr &transposeB,
                             BoolAttr &packedB) {
  Type i1 = p.getBuilder().getIntegerType(1);
  return parseAttrGroup(
      p, {{"transpose_b", [&] { return p.parseAttribute(transposeB, i1); }},
          {"packed_b", [&] { return p.parseAttribute(packedB, i1); }}});
}

void printMatmulAttrs(OpAsmPrinter &p, Operation *, BoolAttr transposeB,
                      BoolAttr packedB) {
  printAttrGroup(p, {nonDefault("transpose_b", transposeB, false),
                     nonDefault("packed_b", packedB, false)});
}

ParseResult parseReduceScatterSumAttrs(OpAsmParser &p, IntegerAttr &axis,
                                       IntegerAttr &groupSize) {
  Type i64 = p.getBuilder().getIntegerType(64);
  return parseAttrGroup(
      p, {{"axis", [&] { return p.parseAttribute(axis, i64); }},
          {"group_size", [&] { return p.parseAttribute(groupSize, i64); }}});
}

void printReduceScatterSumAttrs(OpAsmPrinter &p, Operation *, IntegerAttr axis,
                                IntegerAttr groupSize) {
  printAttrGroup(p, {nonDefault("axis", axis, -1),
                     nonDefault("group_size", groupSize, 0)});
}

ParseResult parseMatmulReduceScatterSumAttrs(OpAsmParser &p,
                                             BoolAttr &hasResidual,
                                             IntegerAttr &residualPeer) {
  Builder &b = p.getBuilder();
  return parseAttrGroup(
      p, {{"has_residual",
           [&] { return p.parseAttribute(hasResidual, b.getIntegerType(1)); }},
          {"residual_peer", [&] {
             return p.parseAttribute(residualPeer, b.getIntegerType(64));
           }}});
}

void printMatmulReduceScatterSumAttrs(OpAsmPrinter &p, Operation *,
                                      BoolAttr hasResidual,
                                      IntegerAttr residualPeer) {
  printAttrGroup(p, {nonDefault("has_residual", hasResidual, false),
                     nonDefault("residual_peer", residualPeer, 0)});
}

ParseResult parseReduceScatterRMSNormAttrs(OpAsmParser &p,
                                           IntegerAttr &groupSize,
                                           BoolAttr &hasResidual) {
  Builder &b = p.getBuilder();
  return parseAttrGroup(
      p, {{"group_size",
           [&] { return p.parseAttribute(groupSize, b.getIntegerType(64)); }},
          {"has_residual", [&] {
             return p.parseAttribute(hasResidual, b.getIntegerType(1));
           }}});
}

void printReduceScatterRMSNormAttrs(OpAsmPrinter &p, Operation *,
                                    IntegerAttr groupSize,
                                    BoolAttr hasResidual) {
  printAttrGroup(p, {nonDefault("group_size", groupSize, 0),
                     nonDefault("has_residual", hasResidual, false)});
}

ParseResult parseAllGatherRMSNormQuantMXFP6Attrs(OpAsmParser &p,
                                                 IntegerAttr &groupSize,
                                                 IntegerAttr &fp6Format) {
  Type i64 = p.getBuilder().getIntegerType(64);
  return parseAttrGroup(
      p, {{"group_size", [&] { return p.parseAttribute(groupSize, i64); }},
          {"fp6_format", [&] { return p.parseAttribute(fp6Format, i64); }}});
}

void printAllGatherRMSNormQuantMXFP6Attrs(OpAsmPrinter &p, Operation *,
                                          IntegerAttr groupSize,
                                          IntegerAttr fp6Format) {
  printAttrGroup(p, {nonDefault("group_size", groupSize, 0),
                     nonDefault("fp6_format", fp6Format, 0)});
}

ParseResult parseGroupedMatmulSwigluNvfp4Attrs(OpAsmParser &p,
                                               StringAttr &activation,
                                               ElementsAttr &swigluAlpha,
                                               ElementsAttr &swigluLimit,
                                               BoolAttr &hasARowScales) {
  Type i1 = p.getBuilder().getIntegerType(1);
  return parseAttrGroup(
      p, {{"activation", [&] { return p.parseAttribute(activation); }},
          {"swiglu_alpha", [&] { return p.parseAttribute(swigluAlpha); }},
          {"swiglu_limit", [&] { return p.parseAttribute(swigluLimit); }},
          {"has_a_row_scales",
           [&] { return p.parseAttribute(hasARowScales, i1); }}});
}

void printGroupedMatmulSwigluNvfp4Attrs(OpAsmPrinter &p, Operation *,
                                        StringAttr activation,
                                        ElementsAttr swigluAlpha,
                                        ElementsAttr swigluLimit,
                                        BoolAttr hasARowScales) {
  printAttrGroup(p, {nonDefault("activation", activation, "silu"),
                     {"swiglu_alpha", swigluAlpha},
                     {"swiglu_limit", swigluLimit},
                     nonDefault("has_a_row_scales", hasARowScales, false)});
}

ParseResult parseCustomAttrs(OpAsmParser &p, StringAttr &symbol,
                             DeviceRefAttr &device,
                             DictionaryAttr &parameters) {
  return parseAttrGroup(
      p, {{"symbol", [&] { return p.parseAttribute(symbol); }, true},
          {"device", [&] { return p.parseAttribute(device); }},
          {"parameters", [&] { return p.parseAttribute(parameters); }}});
}

void printCustomAttrs(OpAsmPrinter &p, Operation *op, StringAttr symbol,
                      DeviceRefAttr device, DictionaryAttr parameters) {
  MLIRContext *context = op->getContext();
  Attribute setDevice;
  if (device && device != DeviceRefAttr::get(context, "cpu", 0))
    setDevice = device;
  Attribute setParameters;
  if (parameters && !parameters.empty())
    setParameters = parameters;
  printAttrGroup(p, {{"symbol", symbol},
                     {"device", setDevice},
                     {"parameters", setParameters}});
}

ParseResult parseConstantExternalAttrs(OpAsmParser &p, StringAttr &name,
                                       IntegerAttr &align,
                                       DeviceRefAttr &device,
                                       BoolAttr &hasAlias,
                                       BoolAttr &isPlaceholder) {
  Builder &b = p.getBuilder();
  return parseAttrGroup(
      p, {{"name", [&] { return p.parseAttribute(name); }, true},
          {"align",
           [&] {
             return p.parseAttribute(align,
                                     b.getIntegerType(64, /*isSigned=*/false));
           },
           true},
          {"device", [&] { return p.parseAttribute(device); }, true},
          {"hasAlias",
           [&] { return p.parseAttribute(hasAlias, b.getIntegerType(1)); }},
          {"isPlaceholder", [&] {
             return p.parseAttribute(isPlaceholder, b.getIntegerType(1));
           }}});
}

void printConstantExternalAttrs(OpAsmPrinter &p, Operation *, StringAttr name,
                                IntegerAttr align, DeviceRefAttr device,
                                BoolAttr hasAlias, BoolAttr isPlaceholder) {
  Attribute setHasAlias;
  if (hasAlias && hasAlias.getValue())
    setHasAlias = hasAlias;
  Attribute setIsPlaceholder;
  if (isPlaceholder && isPlaceholder.getValue())
    setIsPlaceholder = isPlaceholder;
  printAttrGroup(p, {{"name", name},
                     {"align", align},
                     {"device", device},
                     {"hasAlias", setHasAlias},
                     {"isPlaceholder", setIsPlaceholder}});
}

ParseResult parseCallAttrs(OpAsmParser &p, StringAttr &prefix,
                           ArrayAttr &argAttrs, ArrayAttr &resAttrs) {
  return parseAttrGroup(
      p, {{"prefix", [&] { return p.parseAttribute(prefix); }},
          {"arg_attrs", [&] { return p.parseAttribute(argAttrs); }},
          {"res_attrs", [&] { return p.parseAttribute(resAttrs); }}});
}

void printCallAttrs(OpAsmPrinter &p, Operation *, StringAttr prefix,
                    ArrayAttr argAttrs, ArrayAttr resAttrs) {
  Attribute setPrefix;
  if (prefix && !prefix.getValue().empty())
    setPrefix = prefix;
  printAttrGroup(p, {{"prefix", setPrefix},
                     {"arg_attrs", argAttrs},
                     {"res_attrs", resAttrs}});
}

ParseResult parseShapeFuncAttrs(OpAsmParser &p, DictionaryAttr &kgenParams,
                                ArrayAttr &dataDeptTensors) {
  return parseAttrGroup(
      p,
      {{"kgenParams", [&] { return p.parseAttribute(kgenParams); }},
       {"dataDeptTensors", [&] { return p.parseAttribute(dataDeptTensors); }}});
}

void printShapeFuncAttrs(OpAsmPrinter &p, Operation *,
                         DictionaryAttr kgenParams, ArrayAttr dataDeptTensors) {
  printAttrGroup(
      p, {{"kgenParams", kgenParams}, {"dataDeptTensors", dataDeptTensors}});
}

/// Parse the body region of `mo.sequence`: `(%arg0, %arg1) -> (t0, t1) {...}`.
/// Block-argument types are omitted in the syntax; they are 1:1 with, and
/// identically typed to, the operands, whose types arrive via `inputTypes`.
/// The optional `-> (...)` clause populates the op's result types.
ParseResult parseSequenceRegion(OpAsmParser &p, Region &region,
                                TypeRange inputTypes,
                                SmallVectorImpl<Type> &resultTypes) {
  SmallVector<OpAsmParser::Argument> args;
  if (p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/false))
    return failure();
  if (args.size() != inputTypes.size())
    return p.emitError(p.getCurrentLocation(), "expected ")
           << inputTypes.size() << " block argument(s) (one per input), got "
           << args.size();
  for (auto [arg, type] : llvm::zip(args, inputTypes))
    arg.type = type;

  if (succeeded(p.parseOptionalArrow()))
    if (p.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                  [&]() -> ParseResult {
                                    Type type;
                                    if (p.parseType(type))
                                      return failure();
                                    resultTypes.push_back(type);
                                    return success();
                                  }))
      return failure();

  return p.parseRegion(region, args, /*enableNameShadowing=*/false);
}

/// Print the body region of `mo.sequence`. Block-argument names are printed
/// without types (they mirror the operands).
void printSequenceRegion(OpAsmPrinter &p, Operation *op, Region &region,
                         TypeRange inputTypes, TypeRange resultTypes) {
  Block &body = region.front();
  p << '(';
  llvm::interleaveComma(body.getArguments(), p, [&](BlockArgument arg) {
    p.printRegionArgument(arg, /*argAttrs=*/{}, /*omitType=*/true);
  });
  p << ')';
  if (!resultTypes.empty()) {
    p << " -> (";
    llvm::interleaveComma(resultTypes, p,
                          [&](Type type) { p.printType(type); });
    p << ')';
  }
  p << ' ';
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

} // namespace M::MO

SmallVector<ConstantOp> M::MO::tryGetConstantOpsOfRank(TensorValue val,
                                                       size_t rank) {
  SmallVector<ConstantOp> constantOps;
  Operation *inputOp = val.getDefiningOp();

  // Handle identity chains.
  auto isIdentityLike = [&](Operation *op) {
    if (isa_and_nonnull<RebindOp>(op))
      return true;
    if (isa_and_nonnull<mlir::UnrealizedConversionCastOp>(op))
      return true;
    if (isa_and_nonnull<ReshapeOp>(op)) {
      // After a ReshapeOp, accept constants of any rank.
      rank = std::numeric_limits<size_t>::max();
      return true;
    }
    return false;
  };

  // TODO(#37571)
  while (isIdentityLike(inputOp))
    inputOp = inputOp->getOperand(0).getDefiningOp();

  // Handle concatenations of constant tensors.
  if (auto concatOp = dyn_cast_if_present<ConcatOp>(inputOp)) {
    for (Value operand : concatOp.getInputs()) {
      TensorValue tensorOperand = cast<TensorValue>(operand);
      SmallVector<ConstantOp> operandConstantOps =
          tryGetConstantOpsOfRank(tensorOperand, rank);
      if (operandConstantOps.empty())
        return {};
      llvm::append_range(constantOps, operandConstantOps);
    }
    return constantOps;
  }

  auto constantOp = dyn_cast_if_present<ConstantOp>(inputOp);
  if (!constantOp)
    return {};

  MO::TensorType type = constantOp.getType();
  if (rank != std::numeric_limits<size_t>::max() &&
      (!type.hasRank() || type.getRank() != rank))
    return {};

  constantOps.push_back(constantOp);

  return constantOps;
}

std::optional<float> M::MO::tryGetConstantFloat(TensorValue val) {
  if (auto maybeFloatBlob = tryGetConstantOfRank<float>(val, 0))
    return maybeFloatBlob.value().front();
  return std::nullopt;
}

std::optional<double> M::MO::tryGetConstantDouble(TensorValue val) {
  if (auto maybeDoubleBlob = tryGetConstantOfRank<double>(val, 0))
    return maybeDoubleBlob.value().front();
  return std::nullopt;
}

std::optional<int64_t> M::MO::tryGetConstantInt(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<int64_t>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<bool> M::MO::tryGetConstantBool(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<int64_t>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<APFloat> M::MO::tryGetConstantAPFloat(TensorValue val) {
  if (auto maybeBlob = tryGetConstantOfRank<APFloat>(val, 0))
    return maybeBlob.value().front();
  return std::nullopt;
}

std::optional<APInt> M::MO::tryGetConstantAPInt(TensorValue val) {
  if (auto maybeBlob = tryGetConstantOfRank<APInt>(val, 0))
    return maybeBlob.value().front();
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
M::MO::tryGetConstantIntVector(TensorValue val) {
  if (val.getType().getElementType().isIntOrIndex())
    return tryGetConstantOfRank<int64_t>(val, 1);
  return std::nullopt;
}

std::optional<SmallVector<bool>>
M::MO::tryGetConstantBoolVector(TensorValue val) {
  if (val.getType().getElementType().isInteger(1))
    return tryGetConstantOfRank<bool>(val, 1);
  return std::nullopt;
}

/// Returns the `MOSH::ShapeType` value that `val` (an `MO::TensorType`) was
/// created from, or null.
static Value tryGetShapeValue(Value val) {
  // Skip any number of rebinds.
  while (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto toTensorOp = val.getDefiningOp<Shape::ToTensorOp>())
    return toTensorOp.getInput();

  return {};
}

/// Returns a `ShapeAttr` of the constant integers that created `val` (an
/// `MO::TensorType`), or null.
static ShapeAttr tryGetConstantShapeAttr(Value val) {
  // Skip at most a single intermediate rebind
  if (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto tensorVal = dyn_cast<TensorValue>(val)) {
    if (auto maybeIntVec = tryGetConstantIntVector(tensorVal)) {
      MLIRContext *ctx = val.getContext();
      auto shapeType = ShapeType::get(ctx);
      return ShapeAttr::get(maybeIntVec.value(), shapeType);
    }
  }
  return {};
}

ShapeAttr M::MO::tryGetShapeAttr(Value val) {
  if (Value shapeVal = tryGetShapeValue(val)) {
    if (auto paramToValueOp = shapeVal.getDefiningOp<MOSH::ParamToValueOp>())
      return dyn_cast<ShapeAttr>(paramToValueOp.getValue());

    return {};
  }

  return tryGetConstantShapeAttr(val);
}

TypedAttr M::MO::tryGetDimAttr(Value val) {
  auto tensorVal = dyn_cast<TensorValue>(val);
  if (!tensorVal)
    return {};

  if (TypedAttr param = getParamAttrThrough<IndexToTensorOp>(tensorVal))
    return param;

  std::optional<int64_t> maybeConst = tryGetConstantInt(tensorVal);
  if (!maybeConst.has_value())
    return {};

  return MOSH::ShapeAttr::getDim(val.getContext(), *maybeConst);
}

/// Verifies that the first block of the given `region` is terminated by a
/// TerminatorTy. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(Operation *op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = nullptr;
  if (!region.empty() && !region.front().empty()) {
    terminatorOperation = &region.front().back();
    if (auto yield = dyn_cast_if_present<TerminatorTy>(terminatorOperation))
      return yield;
  }
  auto diag = op->emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
}

void M::MO::synchronizeMetadataUpdates(GraphOp op) {
  MLIRContext *ctx = op.getContext();
  TypeRange newArgTypes = op.getBody().getArgumentTypes();
  SmallVector<Type> newResTypes;

  for (auto [newOutTy, funcOutTy] :
       llvm::zip(op.getOutputOp().getOperandTypes(), op.getResultTypes())) {
    // By default, assume nothing changed.
    Type newResTy = funcOutTy;

    if (auto funcOutTensorTy = dyn_cast<TensorType>(funcOutTy)) {
      auto newOutTensorTy = cast<TensorType>(newOutTy);
      newResTy = TensorType::get(
          funcOutTensorTy.getShape(), funcOutTensorTy.getDtype(),
          newOutTensorTy.getDeviceRef(), newOutTensorTy.getMetadata());
    }

    if (auto funcOutBufferTy = dyn_cast<BufferType>(funcOutTy)) {
      auto newOutBufferTy = cast<BufferType>(newOutTy);

      newResTy = BufferType::get(
          funcOutBufferTy.getShape(), funcOutBufferTy.getDtype(),
          newOutBufferTy.getDeviceRef(), newOutBufferTy.getMetadata());
    }

    if (auto funcOutListTy = dyn_cast<ListType>(funcOutTy)) {
      auto newOutListTy = cast<ListType>(newOutTy);
      newResTy = ListType::get(funcOutListTy.getContext(),
                               newOutListTy.getElementType());
    }

    newResTypes.push_back(newResTy);
  }

  auto funcType = FunctionType::get(ctx, newArgTypes, newResTypes);
  FuncTypeGeneratorType signature =
      KGEN::FuncTypeGeneratorType::remapToFuncTypeGenerator(
          op.getInputParameters(), funcType);
  op.setSignature(signature);
  op.setFunctionType(funcType);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto graphOp =
      symbolTable.lookupNearestSymbolFrom<GraphOp>(*this, getCalleeAttr());
  if (!graphOp)
    return emitError("'") << getCalleeAttr()
                          << "' does not reference a valid graph";

  if (!graphOp.getIsSubgraph())
    return emitError() << "Only subgraphs can be called";

  auto fnType = graphOp.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitError() << "Subgraph " << graphOp.getName()
                       << " has wrong number of arguments";

  auto emitErrorFn = [&](size_t idx, Type graphArgType, Type operandType) {
    auto diag = emitError() << "Subgraph " << graphOp.getName()
                            << " has wrong type for argument " << idx
                            << " (function type: " << graphArgType
                            << ", operand type: " << operandType << ")";
    diag.attachNote(graphOp.getLoc()) << "graph op here";
    return failure();
  };

  for (auto [idx, graphArgType, operandType] : llvm::enumerate(
           graphOp.getFunctionType().getInputs(), getOperandTypes())) {
    if (isa<TensorType>(graphArgType) && isa<TensorType>(operandType)) {
      auto graphArgTensorType = cast<TensorType>(graphArgType);
      auto operandTensorType = cast<TensorType>(operandType);
      if (graphArgTensorType.cloneWithoutLayout() !=
          operandTensorType.cloneWithoutLayout()) {
        return emitErrorFn(idx, graphArgType, operandType);
      }
    } else if (graphArgType != operandType) {
      return emitErrorFn(idx, graphArgType, operandType);
    }
  }

  return success();
}

LogicalResult CallOp::verify() {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();

  // Add this logic here to ensure eager verification of call operations
  // at build time in the Graph API.
  mlir::SymbolTableAnalysis symTabAnalysis(moduleOp);
  return verifySymbolUses(symTabAnalysis.getSymbolTables());
}

bool CallOp::isInPlace() {
  return !getInChains().empty() && !getOutChains().empty();
}

SmallVector<ChainValue> CallOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> CallOp::getOutChains() {
  return getChainValues(getResults());
}

SmallVector<OpOperand *> CallOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getOperandsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

void TransferOp::build(OpBuilder &builder, OperationState &state,
                       TensorValue inputValue, DeviceRefAttr destDevice,
                       ChainValue inChain, bool alwaysElideSameDeviceCopy) {
  auto resultTy = inputValue.getType().cloneWith(destDevice);
  auto chainTy = ChainType::get(builder.getContext());
  build(builder, state, resultTy, chainTy, inputValue,
        alwaysElideSameDeviceCopy, inChain);
}

LogicalResult TransferOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType resultType = getResult().getType();
  TensorType resTypeWithInputDevice =
      resultType.cloneWith(inputType.getDeviceRef());
  if (resTypeWithInputDevice != inputType) {
    return emitError()
           << "expected result type to only differ from input type in device ("
           << inputType << " vs " << resultType << ").";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StableCreateOp
//===----------------------------------------------------------------------===//

LogicalResult StableCreateOp::verify() {
  if (!getResult().getType().isStable())
    return emitOpError("result must be marked `stable`");
  return success();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

/// Name of optional attribute holding the #M.device_spec_collection describing
/// the (virtual) devices available to the graph.
constexpr StringRef kGraphDeviceSpecCollectionAttrName = "device_specs";

constexpr StringRef kGraphTargetInfoAttrName = "target_info";

/// Traverse the given types and collect unique parameter references, then
/// return a set of parameter declaration for those references.
static llvm::SetVector<KGEN::ParamDeclAttr>
getParamDeclsFromShapes(TypeRange types) {
  llvm::SetVector<KGEN::ParamDeclAttr> uniqueDecls;
  for (Type type : types) {
    type.walk([&](KGEN::ParamDeclRefAttr declRef) {
      uniqueDecls.insert(KGEN::ParamDeclAttr::get(declRef));
    });
  }
  return uniqueDecls;
}

void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                    TypeRange inputTypes, TypeRange resultTypes,
                    bool isSubgraph) {
  // Specializes `buildWithEntryBlock`, which doesn't work because we use
  // `signature` and not `function_type`.
  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));
  if (isSubgraph) {
    state.addAttribute(getIsSubgraphAttrName(state.name),
                       builder.getUnitAttr());
  }
  FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto sig = KGEN::FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, funcType);
  state.addAttribute(getSignatureAttrName(state.name), TypeAttr::get(sig));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(funcType));

  // Declare input/result parameters, if any is used in the input/result types.
  MLIRContext *ctx = builder.getContext();
  auto inputDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(inputTypes).getArrayRef());
  auto resultDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(resultTypes).getArrayRef());
  state.addAttribute(getInputParametersAttrName(state.name), inputDecls);
  state.addAttribute(getResultParametersAttrName(state.name), resultDecls);

  Region *bodyRegion = state.addRegion();
  Block &body = bodyRegion->emplaceBlock();
  for (Type input : inputTypes)
    body.addArgument(input, state.location);
}

// GraphOp's `counter` property generates unique symbolic dimension names. It
// round-trips as a trailing extra attribute on the graph.
static constexpr StringLiteral GRAPH_OP_COUNTER_NAME = "counter";
static ParseResult parseGraphOp(OpAsmParser &p, StringAttr &visibility,
                                StringAttr &name, TypeAttr &signatureAttr,
                                TypeAttr &functionTypeAttr,
                                KGEN::ParamDeclArrayAttr &inputParams,
                                KGEN::ParamDeclArrayAttr &resultParams,
                                UnitAttr &isSubgraph, UnitAttr &isDeviceGraph,
                                NamedAttrList &attrs, Region &body,
                                int64_t &counter) {
  StringRef visibilityKeyword;
  if (succeeded(p.parseOptionalKeyword(&visibilityKeyword,
                                       {"public", "private", "nested"})))
    visibility = p.getBuilder().getStringAttr(visibilityKeyword);

  if (p.parseSymbolName(name))
    return failure();

  auto parserHook = [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    return parseParamDeclElement(p, decls);
  };

  SmallVector<OpAsmParser::Argument> args;
  KGEN::FuncTypeGeneratorType signature;
  FunctionType functionType;
  if (KGEN::parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                           functionType, signature,
                                           /*parseDeclElt=*/parserHook))
    return failure();
  signatureAttr = TypeAttr::get(signature);
  functionTypeAttr = TypeAttr::get(functionType);

  if (succeeded(p.parseOptionalKeyword("isSubgraph")))
    isSubgraph = p.getBuilder().getUnitAttr();
  if (succeeded(p.parseOptionalKeyword("isDeviceGraph")))
    isDeviceGraph = p.getBuilder().getUnitAttr();

  if (p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();

  NamedAttrList extraAttrs;
  if (p.parseOptionalAttrDict(extraAttrs))
    return failure();

  counter = 0;
  if (!extraAttrs.empty()) {
    ASSERT_STREAM(extraAttrs.getAttrs().size() == 1,
                  << "Expect 'counter' to be only extra attribute.");
    NamedAttribute attr = *extraAttrs.begin();
    ASSERT_STREAM(attr.getName() == GRAPH_OP_COUNTER_NAME,
                  "Expect 'counter' to be only extra attribute.");
    counter = cast<IntegerAttr>(attr.getValue()).getInt();
  }

  return success();
}

static void printGraphOp(OpAsmPrinter &p, Operation *op, StringAttr visibility,
                         StringAttr name, TypeAttr signature,
                         TypeAttr functionType,
                         KGEN::ParamDeclArrayAttr inputParams,
                         KGEN::ParamDeclArrayAttr resultParams,
                         UnitAttr isSubgraph, UnitAttr isDeviceGraph,
                         DictionaryAttr attrs, Region &body, int64_t counter) {
  auto printerHook = [&](KGEN::ParamDeclAttr decl) {
    printParamDeclElement(p, decl);
  };

  if (visibility)
    p << visibility.getValue() << ' ';
  p.printSymbolName(name);
  KGEN::printFunctionFuncTypeGenerator(
      p, &body, inputParams, resultParams,
      cast<FunctionType>(functionType.getValue()),
      cast<FuncTypeGeneratorType>(signature.getValue()),
      /*printInputElt=*/printerHook,
      /*printResultElt=*/printerHook);

  if (isSubgraph)
    p << " isSubgraph";
  if (isDeviceGraph)
    p << " isDeviceGraph";

  auto graph = cast<GraphOp>(op);
  SmallVector<StringRef, 8> elidedAttrs{
      graph.getSymNameAttrName(),         graph.getSymVisibilityAttrName(),
      graph.getSignatureAttrName(),       graph.getFunctionTypeAttrName(),
      graph.getInputParametersAttrName(), graph.getResultParametersAttrName(),
      graph.getIsSubgraphAttrName(),      graph.getIsDeviceGraphAttrName()};
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);

  // Extra metadata at end for use internally in compiler
  SmallVector<NamedAttribute> extraAttrs;
  if (counter != 0) {
    auto counterAttr =
        IntegerAttr::get(IntegerType::get(op->getContext(), 64), counter);
    auto counterAttrName =
        StringAttr::get(op->getContext(), GRAPH_OP_COUNTER_NAME);
    extraAttrs.push_back(NamedAttribute(counterAttrName, counterAttr));
  }
  p.printOptionalAttrDict(extraAttrs);
}

OutputOp GraphOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

LogicalResult GraphOp::verify() {
  KGEN::ParameterCollector::Analysis cache;
  KGEN::ParameterUseDefGraph useDefGraph(getRegion());

  // check that no expressions in the input parameters
  mlir::AttrTypeWalker walker;
  bool found = false;
  walker.addWalk([&](KGEN::ParamOperatorAttr attr) {
    found = true;
    return WalkResult::interrupt();
  });
  for (auto ty : getArgumentTypes()) {
    walker.walk(ty);
    if (found)
      break;
  }
  if (found) {
    return emitError(
        "Expressions are not (currently) allowed in function signatures");
  }

  return useDefGraph.verify(nullptr, cache);
}

KGEN::ParamDeclAttr GraphOp::getNewDimParam(const Twine &prefix) {
  auto dimType = MOSH::ShapeAttr::getDimType(getContext());
  return getNewParamOfType(prefix, dimType);
}

KGEN::ParamDeclAttr GraphOp::getNewParamOfType(const Twine &prefix, Type type) {
  return KGEN::ParamDeclAttr::get(getNewParamName(prefix), type);
}

std::string GraphOp::getNewParamName(const Twine &prefix) {
  // TODO(#8829) avoid duplication with existing parameters in the input graph
  int64_t newCounter = getCounter() + 1;
  setCounter(newCounter);
  return (prefix + Twine(newCounter)).str();
}

void GraphOp::setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs) {
  getOperation()->setAttr(kGraphDeviceSpecCollectionAttrName, deviceSpecs);
}

TargetInfoAttr GraphOp::getDeviceTargetInfo() {
  return dyn_cast_if_present<TargetInfoAttr>(
      getOperation()->getAttr(kGraphTargetInfoAttrName));
}
void GraphOp::setDeviceTargetInfo(TargetInfoAttr targetInfo) {
  getOperation()->setAttr(kGraphTargetInfoAttrName, targetInfo);
}

ArrayRef<ParamDeclAttr> GraphOp::getInputParams() {
  return getInputParameters();
}

SmallVector<ParamDeclAttr> GraphOp::getAllParams() {
  SmallVector<ParamDeclAttr> params;
  params.append(getInputParameters().begin(), getInputParameters().end());
  params.append(getResultParameters().begin(), getResultParameters().end());
  return params;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

/// Parse the output values for `mo.output`, sugaring `?` for unknown values.
static ParseResult parseOutputValues(AsmParser &p,
                                     KGEN::ParameterExprArrayAttr &valuesAttr) {
  SmallVector<TypedAttr> values;
  auto parseFn = [&]() -> ParseResult {
    Type type = ShapeAttr::getDimType(p.getContext());
    if (succeeded(p.parseOptionalColon()) && KGEN::parseKGENType(p, type))
      return failure();
    return KGEN::parseParamValue(p, values.emplace_back(), type);
  };
  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::OptionalLessGreater,
                                parseFn))
    return failure();
  valuesAttr = ParameterExprArrayAttr::get(p.getContext(), values);
  return success();
}

/// Print the output values for `mo.output`, sugaring `?` for unknown values.
static void printOutputValues(AsmPrinter &p, Operation *op,
                              ArrayRef<TypedAttr> values) {
  if (values.empty())
    return;
  p << '<';
  llvm::interleaveComma(values, p, [&](TypedAttr value) {
    if (Type type = value.getType(); !ShapeAttr::isDimType(type)) {
      p << ':';
      KGEN::printKGENType(p, type);
      p << ' ';
    }
    KGEN::printParamValue(p, value);
  });
  p << '>';
}

void OutputOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParameters();
  ArrayRef<TypedAttr> defs = this->getParameters();

  // Doesn't make sense to walk, graph is in a bad state.
  if (decls.size() != defs.size()) {
    emitError("[OutputOp::walkDefinitions] output bindings and result "
              "parameters must have the same size");
    return;
  }

  for (auto [decl, def] : llvm::zip(decls, defs))
    walkDef(decl, def);
}

LogicalResult OutputOp::verify() {
  // ODS checks the parent is a GraphOp.
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParameters();
  ArrayRef<TypedAttr> defs = this->getParameters();
  if (decls.size() != defs.size()) {
    return emitError(
        "output bindings and result parameters must have the same size");
  }

  // Build a mapping from result param references to their binding attribute,
  // example:
  //  { (D1, 42), (Sh0, [D0, 10]) }
  DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> resParamRefToBindingAttr;
  for (auto [resParamDecl, bindingParam] : llvm::zip(decls, defs)) {
    resParamRefToBindingAttr.insert(
        {KGEN::ParamDeclRefAttr::get(resParamDecl), bindingParam});
  }

  // Substitute param references inside graph result types using the map, which
  // gives us the expected output operands type.
  SmallVector<Type> expectedOutputOperandTypes;
  for (Type ty : graph.getResultTypes()) {
    if (auto tensorTy = dyn_cast<TensorType>(ty)) {
      auto newShape = cast<MOSH::ShapeAttr>(tensorTy.getShape().replace(
          [&resParamRefToBindingAttr](KGEN::ParamDeclRefAttr paramRef) {
            TypedAttr resolvedAttr = resParamRefToBindingAttr.lookup(paramRef);
            return resolvedAttr ? resolvedAttr : paramRef;
          }));
      expectedOutputOperandTypes.emplace_back(tensorTy.cloneWith(newShape));
    } else {
      expectedOutputOperandTypes.emplace_back(ty);
    }
  }

  return checkOperandTypes(getOperation(), expectedOutputOperandTypes);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  Operation *parentOp = getOperation()->getParentOp();
  if (auto ifOp = dyn_cast<IfOp>(parentOp)) {
    ArrayRef<KGEN::ParamDeclAttr> decls = ifOp.getOutputParamDecls();
    ArrayRef<TypedAttr> defs = this->getParameters();
    if (decls.size() != defs.size()) {
      return emitOpError(
                 "[YieldOp::walkDefinitions] yield parameter bindings (")
             << defs.size() << ") and if output parameters (" << decls.size()
             << ") must have the same size";
    }
  }
  return success();
}

void YieldOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {

  Operation *parentOp = getOperation()->getParentOp();

  if (auto ifOp = dyn_cast<IfOp>(parentOp)) {
    ArrayRef<KGEN::ParamDeclAttr> decls = ifOp.getOutputParamDecls();
    ArrayRef<TypedAttr> defs = this->getParameters();
    for (auto [decl, def] : llvm::zip(decls, defs))
      walkDef(decl, def);
  }
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> IfOp::getInputParams() { return {}; }
SmallVector<ParamDeclAttr> IfOp::getAllParams() {
  SmallVector<ParamDeclAttr> params;
  params.append(getOutputParamDecls().begin(), getOutputParamDecls().end());
  return params;
}

Block *IfOp::getThenBlock() { return &getThenRegion().front(); }

Block *IfOp::getElseBlock() { return &getElseRegion().front(); }

void IfOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.emplace_back(mlir::RegionSuccessor(getOperation()));
    return;
  }

  regions.emplace_back(&getThenRegion());
  regions.emplace_back(&getElseRegion());
}

ValueRange IfOp::getSuccessorInputs(mlir::RegionSuccessor successor) {
  return successor.isOperation() ? getOperation()->getResults() : ValueRange();
}

void IfOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  regions.emplace_back(&getThenRegion());
  regions.emplace_back(&getElseRegion());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> WhileOp::getInputParams() { return {}; }
SmallVector<ParamDeclAttr> WhileOp::getAllParams() { return {}; }

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  Region *condRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();

  if (parseParenOperandListWithShadowing(parser, result, argumentInfo,
                                         /*isIsolatedFromAbove=*/false) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegion(parser, argumentInfo, *condRegion,
                  /*isIsolatedFromAbove=*/false) ||
      parser.parseKeyword("do") ||
      parseRegion(parser, argumentInfo, *bodyRegion,
                  /*isIsolatedFromAbove=*/false))
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
                                     /*isIsolatedFromAbove=*/false);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegion(p, getInputs(), getCondRegion(), /*isIsolatedFromAbove=*/false);
  p << " do ";
  printRegion(p, getInputs(), getBodyRegion(), /*isIsolatedFromAbove=*/false);
}

OperandRange
WhileOp::getEntrySuccessorOperands(mlir::RegionSuccessor successor) {
  assert(successor.getSuccessor() == &getCondRegion() &&
         "WhileOp is expected to branch only to the first region");
  return getOperands();
}

void WhileOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (point.isParent()) {
    regions.emplace_back(&getCondRegion());
    return;
  }

  assert(llvm::is_contained(
             {&getCondRegion(), &getBodyRegion()},
             point.getTerminatorPredecessorOrNull()->getParentRegion()) &&
         "there are only two regions in a WhileOp");
  // The body region always branches back to the condition region.
  if (point.getTerminatorPredecessorOrNull()->getParentRegion() ==
      &getBodyRegion()) {
    regions.emplace_back(&getCondRegion());
    return;
  }

  regions.emplace_back(mlir::RegionSuccessor(getOperation()));
  regions.emplace_back(&getBodyRegion());
}

ValueRange WhileOp::getSuccessorInputs(mlir::RegionSuccessor successor) {
  if (successor.isOperation())
    return getOperation()->getResults();
  if (successor == &getCondRegion())
    return getCondRegion().getArguments();
  if (successor == &getBodyRegion())
    return getBodyRegion().getArguments();
  llvm_unreachable("invalid region successor");
}

MutableArrayRef<OpOperand> WhileOp::getInitsMutable() {
  return getInputsMutable();
}

std::optional<MutableArrayRef<OpOperand>> WhileOp::getYieldedValuesMutable() {
  return getBodyTerminator().getOperandsMutable();
}

Block::BlockArgListType WhileOp::getRegionIterArgs() {
  return getCondRegion().getArguments();
}

SmallVector<Region *> WhileOp::getLoopRegions() {
  return {&getCondRegion(), &getBodyRegion()};
}

LogicalResult WhileOp::verify() {
  auto beforeTerminator = verifyAndGetTerminator<WhileConditionOp>(
      *this, getCondRegion(),
      "expects the 'before' region to terminate with 'mo.while.condition'");
  if (!beforeTerminator)
    return failure();

  auto afterTerminator = verifyAndGetTerminator<YieldOp>(
      *this, getBodyRegion(),
      "expects the 'after' region to terminate with 'mo.yield'");
  return success(afterTerminator != nullptr);
}

WhileConditionOp WhileOp::getCondTerminator() {
  return cast<WhileConditionOp>(getCondRegion().front().getTerminator());
}

YieldOp WhileOp::getBodyTerminator() {
  return cast<YieldOp>(getBodyRegion().front().getTerminator());
}

Block *WhileOp::getCondBlock() { return &getCondRegion().front(); }

Block *WhileOp::getBodyBlock() { return &getBodyRegion().front(); }

Block::BlockArgListType WhileOp::getCondArguments() {
  return getCondRegion().getArguments();
}

Block::BlockArgListType WhileOp::getBodyArguments() {
  return getBodyRegion().getArguments();
}

SmallVector<ChainValue> WhileOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> WhileOp::getOutChains() {
  return getChainValues(getResults());
}

SmallVector<OpOperand *> WhileOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getInputsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

//===----------------------------------------------------------------------===//
// WhileConditionOp
//===----------------------------------------------------------------------===//

void WhileConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  WhileOp whileOp = getParentOp();

  // Branches to the after region if true, back to the parent op otherwise.
  regions.emplace_back(&whileOp.getBodyRegion());
  regions.emplace_back(mlir::RegionSuccessor(whileOp.getOperation()));
}

MutableOperandRange
WhileConditionOp::getMutableSuccessorOperands(mlir::RegionSuccessor successor) {
  assert((successor.isOperation() ||
          successor.getSuccessor() == &getParentOp().getBodyRegion()) &&
         "condition op can only exit the loop or branch to the after"
         "region");
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

//===----------------------------------------------------------------------===//
// Bundle / Unbundle ops
//===----------------------------------------------------------------------===//

LogicalResult TensorBundleOp::verify() {
  BundleType resultType = getResult().getType();
  if (getInputs().size() != resultType.getNumElements())
    return emitOpError("input count (")
           << getInputs().size() << ") must match bundle element count ("
           << resultType.getNumElements() << ")";
  for (auto [i, input, expected] :
       llvm::enumerate(getInputs(), resultType.getElementTypes()))
    if (input.getType() != expected)
      return emitOpError("input #")
             << i << " type " << input.getType()
             << " does not match bundle element type " << expected;
  return success();
}

LogicalResult TensorUnbundleOp::verify() {
  BundleType inputType = getInput().getType();
  if (getOutputs().getTypes().size() != inputType.getNumElements())
    return emitOpError("output count (")
           << getOutputs().getTypes().size()
           << ") must match bundle element count ("
           << inputType.getNumElements() << ")";
  for (auto [i, output, expected] :
       llvm::enumerate(getOutputs().getTypes(), inputType.getElementTypes()))
    if (output != expected)
      return emitOpError("output #")
             << i << " type " << output
             << " does not match bundle element type " << expected;
  return success();
}

void TensorBundleOp::build(OpBuilder &builder, OperationState &state,
                           ValueRange inputs) {
  SmallVector<Type> elementTypes =
      llvm::map_to_vector(inputs, [](Value v) { return v.getType(); });
  build(builder, state, BundleType::get(builder.getContext(), elementTypes),
        inputs);
}

void TensorUnbundleOp::build(OpBuilder &builder, OperationState &state,
                             Value input) {
  build(builder, state, cast<BundleType>(input.getType()).getElementTypes(),
        input);
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

namespace {

// Returns success if `op` is allowed inside `mo.parallel`, else emits a
// diagnostic. Allowed ops are:
//
//  - Computational ops that lower to a MOGG kernel via `MOToMOGG` (caught
//    here by exclusion from the rejection arms below).
//  - Per-launch entry points for bundled collectives (`mo.bundled.expand`,
//    `mo.bundled.allreduce.sum`), tagged with `MO_BundledCollectiveInterface`.
LogicalResult verifyParallelBodyOp(Operation &op, ParallelOp parallelOp) {
  // Bundled collective ops are only supported within parallel regions.
  // They may contain mutable signal buffers.
  if (isa<MOBundledCollectiveInterface>(&op))
    return success();

  // HACK (see GEX-3701):
  // Prior to HoistParameterExpressionsPass, rebinds may exist in the parallel
  // body. We currently assume they will be moved out before lowering.
  if (isa<MO::RebindOp>(&op))
    return success();

  // mo.buffer is not supported in parallel regions, except signal buffers in
  // bundled collectives and conditionally-in-place ops (`mo.custom`) that are
  // not in-place, which lower to kernels like any computational op.
  if (isa<MOMutableOpInterface>(&op)) {
    auto cond = dyn_cast<MOConditionallyInPlaceInterface>(&op);
    if (!cond || cond.isInPlace())
      return op.emitOpError()
             << "is not allowed inside mo.parallel: mutable buffer ops and "
                "multi-device collectives are not supported; see GEX-3704";
  }

  // `MO_PreservedDuringKernelLowering` tags ops the kernel-lowering pass leaves
  // in place (constants, control flow, transfers, host-only, bundle /
  // unbundle, nested mo.parallel, ...). None are allowed in the body yet.
  if (isa<MO::PreservedDuringKernelLowering>(&op))
    return op.emitOpError()
           << "is not allowed inside mo.parallel: only kernels and bundled "
              "collectives are currently supported in the body; see "
              "GEX-3701, GEX-3704, GEX-3705";

  return success();
}

} // namespace

ArrayRef<ParamDeclAttr> ParallelOp::getInputParams() { return {}; }
SmallVector<ParamDeclAttr> ParallelOp::getAllParams() { return {}; }

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, TypeRange resultTypes) {
  build(builder, state, inputs, /*buffers=*/{}, /*inChain=*/Value{},
        resultTypes);
}

/// Derives the per-launch device list from each bundle element's deviceRef.
/// `BundleType::verify` guarantees every element is a TensorType.
static SmallVector<DeviceRefAttr>
deriveLaunchDevicesFromBundle(BundleType bundle) {
  return llvm::map_to_vector(bundle.getElementTypes(), [](Type t) {
    return cast<TensorType>(t).getDeviceRef();
  });
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, ValueRange buffers, Value inChain,
                       TypeRange resultTypes) {
  assert(!inputs.empty() &&
         "auto-derived ParallelOp::build requires at least one input bundle; "
         "use the explicit-launchDevices overload for bundle-less parallels");
  build(builder, state, inputs, buffers, inChain, resultTypes,
        deriveLaunchDevicesFromBundle(
            cast<BundleType>(inputs.front().getType())));
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, ValueRange buffers, Value inChain,
                       TypeRange resultTypes,
                       ArrayRef<DeviceRefAttr> launchDevices) {
  assert(buffers.empty() == !inChain &&
         "buffers and inChain must be both present or both absent");
  state.addOperands(inputs);
  state.addOperands(buffers);
  if (inChain)
    state.addOperands(inChain);
  state.addTypes(resultTypes);
  state.addAttribute(
      ParallelOp::getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                                    static_cast<int32_t>(buffers.size()),
                                    static_cast<int32_t>(inChain ? 1 : 0)}));
  state.addAttribute(
      ParallelOp::getLaunchDevicesAttrName(state.name),
      builder.getArrayAttr(llvm::map_to_vector(
          launchDevices, [](DeviceRefAttr d) -> Attribute { return d; })));
  // Block args are the first element type of each bundle input, plus a
  // trailing !mo.chain when chained: the in-chain enters the body through it
  // (not captured), and the body threads it through its buffer ops and yields
  // the resulting chain.
  Block &block = state.addRegion()->emplaceBlock();
  SmallVector<Type> argTypes;
  for (Value input : inputs)
    argTypes.push_back(
        cast<BundleType>(input.getType()).getElementTypes().front());
  if (inChain)
    argTypes.push_back(ChainType::get(builder.getContext()));
  SmallVector<Location> locs(argTypes.size(), state.location);
  block.addArguments(argTypes, locs);
}

/// Parse block arguments: `()`, `(%arg0)`, or `(%arg0, %arg1, ...)`.
static ParseResult
parseBlockArgs(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::Argument> &blockArgs) {
  if (parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  do {
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/false))
      return failure();
    blockArgs.push_back(arg);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRParen())
    return failure();
  return success();
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *bodyRegion = result.addRegion();

  SmallVector<OpAsmParser::Argument> blockArgs;
  if (failed(parseBlockArgs(parser, blockArgs)))
    return failure();

  // Parse `in (%bundle0 : type0, %bundle1 : type1, ...)`.  The list may be
  // empty (`in ()`) when the parallel has no input bundles, in which case
  // the `devices(...)` clause must be present (see below).
  if (parser.parseKeyword("in") || parser.parseLParen())
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> bundleOps;
  SmallVector<Type> bundleTypes;
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::UnresolvedOperand op;
      Type type;
      if (parser.parseOperand(op) || parser.parseColon() ||
          parser.parseType(type))
        return failure();
      bundleOps.push_back(op);
      bundleTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  for (auto [op, type] : llvm::zip(bundleOps, bundleTypes))
    if (parser.resolveOperand(op, type, result.operands))
      return failure();

  // Parse optional `buffers(%buf0 : type0, %buf1 : type1, ...)`.
  SmallVector<OpAsmParser::UnresolvedOperand> bufferOps;
  SmallVector<Type> bufferTypes;
  bool hasBuffers = succeeded(parser.parseOptionalKeyword("buffers"));
  if (hasBuffers) {
    if (parser.parseLParen())
      return failure();
    do {
      OpAsmParser::UnresolvedOperand op;
      Type type;
      if (parser.parseOperand(op) || parser.parseColon() ||
          parser.parseType(type))
        return failure();
      bufferOps.push_back(op);
      bufferTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  for (auto [op, type] : llvm::zip(bufferOps, bufferTypes))
    if (parser.resolveOperand(op, type, result.operands))
      return failure();

  // Parse optional `chain(%ch)`. Must be present iff buffers are present.
  bool hasChain = succeeded(parser.parseOptionalKeyword("chain"));
  if (hasChain != hasBuffers)
    return parser.emitError(parser.getCurrentLocation(),
                            "buffers(...) and chain(...) must be both present "
                            "or both absent");
  if (hasChain) {
    OpAsmParser::UnresolvedOperand chainOp;
    if (parser.parseLParen() || parser.parseOperand(chainOp) ||
        parser.parseRParen())
      return failure();
    if (parser.resolveOperand(chainOp, ChainType::get(parser.getContext()),
                              result.operands))
      return failure();
  }

  // Parse optional `devices(label:firstId[-lastId][, ...])` clause. When
  // absent, devices are derived from `inputs.front()` like the C++ builders;
  // with no input bundles, the clause is required.
  SmallVector<DeviceRefAttr> launchDevices;
  if (succeeded(parser.parseOptionalKeyword("devices"))) {
    if (parser.parseLParen() || parseDeviceList(parser, launchDevices) ||
        parser.parseRParen())
      return failure();
  } else if (!bundleTypes.empty()) {
    auto firstBundle = dyn_cast<BundleType>(bundleTypes.front());
    if (!firstBundle)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected non-empty mo.bundle type");
    launchDevices = deriveLaunchDevicesFromBundle(firstBundle);
  } else {
    return parser.emitError(
        parser.getCurrentLocation(),
        "devices(...) clause is required when there are no input bundles");
  }
  result.addAttribute(
      ParallelOp::getLaunchDevicesAttrName(result.name),
      parser.getBuilder().getArrayAttr(llvm::map_to_vector(
          launchDevices, [](DeviceRefAttr d) -> Attribute { return d; })));

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(bundleOps.size()),
                           static_cast<int32_t>(bufferOps.size()),
                           static_cast<int32_t>(hasChain ? 1 : 0)}));

  // Block args: one per bundle input, plus a trailing !mo.chain arg when
  // chained (the in-chain enters the body through it).
  size_t expectedBlockArgs = bundleOps.size() + (hasChain ? 1 : 0);
  if (blockArgs.size() != expectedBlockArgs)
    return parser.emitError(parser.getCurrentLocation(), "expected ")
           << expectedBlockArgs << " block args (one per bundle input"
           << (hasChain ? " plus a trailing chain arg" : "") << "), got "
           << blockArgs.size();

  for (auto [group, bType] : llvm::enumerate(bundleTypes)) {
    auto bundleTy = dyn_cast<BundleType>(bType);
    if (!bundleTy || bundleTy.getNumElements() == 0)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected non-empty mo.bundle type");
    blockArgs[group].type = bundleTy.getElementTypes()[0];
  }
  if (hasChain)
    blockArgs.back().type = ChainType::get(parser.getContext());

  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                       [&]() -> ParseResult {
                                         Type type;
                                         if (parser.parseType(type))
                                           return failure();
                                         resultTypes.push_back(type);
                                         return success();
                                       }))
      return failure();
  }
  if (hasChain)
    resultTypes.push_back(ChainType::get(parser.getContext()));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (parser.parseRegion(*bodyRegion, blockArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  result.types.append(resultTypes);
  return success();
}

void ParallelOp::print(OpAsmPrinter &p) {
  Block &body = getBodyRegion().front();
  auto inputs = getInputs();
  auto buffers = getBuffers();

  p << " (";
  llvm::interleaveComma(body.getArguments(), p, [&](BlockArgument arg) {
    p.printRegionArgument(arg, /*argAttrs=*/{}, /*omitType=*/true);
  });
  p << ')';

  p << " in (";
  llvm::interleaveComma(inputs, p, [&](Value input) {
    p.printOperand(input);
    p << " : ";
    p.printType(input.getType());
  });
  p << ')';

  if (hasBuffers()) {
    p << " buffers(";
    llvm::interleaveComma(buffers, p, [&](Value buf) {
      p.printOperand(buf);
      p << " : ";
      p.printType(buf.getType());
    });
    p << ')';
  }

  if (hasChain()) {
    p << " chain(";
    p.printOperand(getInChain());
    p << ')';
  }

  // Elide `devices(...)` when it is what the parser would derive from
  // `inputs.front()`. With no inputs there is nothing to derive from.
  SmallVector<DeviceRefAttr> launchDevices = getLaunchDeviceList();
  bool elide = !inputs.empty() &&
               launchDevices == deriveLaunchDevicesFromBundle(
                                    cast<BundleType>(inputs.front().getType()));
  if (!elide) {
    p << " devices(";
    printDeviceList(p, launchDevices);
    p << ')';
  }

  SmallVector<Type> nonChainResultTypes;
  for (auto result : getResults())
    if (!isa<ChainType>(result.getType()))
      nonChainResultTypes.push_back(result.getType());
  if (!nonChainResultTypes.empty()) {
    p << " -> (";
    llvm::interleaveComma(nonChainResultTypes, p,
                          [&](Type type) { p.printType(type); });
    p << ')';
  }

  p.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{"operandSegmentSizes", "launchDevices"});
  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

/// Returns true if device labels match; IDs may differ.
static bool deviceLabelsMatch(DeviceRefAttr a, DeviceRefAttr b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return a.getLabel() == b.getLabel();
}

/// Returns true if two types match ignoring device ID.
static bool typesMatchIgnoringDeviceId(Type a, Type b) {
  if (auto tensorA = dyn_cast<TensorType>(a)) {
    auto tensorB = dyn_cast<TensorType>(b);
    if (!tensorB)
      return false;
    return tensorA.getShape() == tensorB.getShape() &&
           tensorA.getDtype() == tensorB.getDtype() &&
           deviceLabelsMatch(tensorA.getDeviceRef(), tensorB.getDeviceRef());
  }
  if (auto bufferA = dyn_cast<BufferType>(a)) {
    auto bufferB = dyn_cast<BufferType>(b);
    if (!bufferB)
      return false;
    return bufferA.getShape() == bufferB.getShape() &&
           bufferA.getDtype() == bufferB.getDtype() &&
           deviceLabelsMatch(bufferA.getDeviceRef(), bufferB.getDeviceRef());
  }
  return a == b;
}

SmallVector<ChainValue> ParallelOp::getInChains() {
  if (auto chain = getInChain())
    return {cast<ChainValue>(chain)};
  return {};
}

SmallVector<ChainValue> ParallelOp::getOutChains() {
  if (hasChain())
    return {getOutChain()};
  return {};
}

SmallVector<OpOperand *> ParallelOp::getInChainsMutable() {
  SmallVector<OpOperand *> result;
  if (getInChain()) {
    auto mutableRange = getInChainMutable();
    for (auto &operand : mutableRange)
      result.push_back(&operand);
  }
  return result;
}

void ParallelOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  Block &body = getBodyRegion().front();

  SmallVector<mlir::MemoryEffects::EffectInstance> bodyEffects;
  for (Operation &op : body) {
    auto nested = mlir::getEffectsRecursively(&op);
    if (!nested)
      return;
    bodyEffects.append(nested->begin(), nested->end());
  }

  size_t numBundleArgs = getInputs().size();

  for (auto &effect : bodyEffects) {
    Value val = effect.getValue();
    auto blockArg = val ? dyn_cast<BlockArgument>(val) : BlockArgument{};
    if (!blockArg || blockArg.getOwner() != &body) {
      effects.push_back(effect);
      continue;
    }

    unsigned argIdx = blockArg.getArgNumber();
    if (argIdx < numBundleArgs) {
      // Remap bundle block arg effects to the bundle operand. If the bundle
      // is defined by a tensor.bundle op, trace through to individual
      // tensor operands.
      Value bundleVal = getInputs()[argIdx];
      Operation *defOp = bundleVal.getDefiningOp();
      if (defOp && isa<TensorBundleOp>(defOp)) {
        for (auto &operand : defOp->getOpOperands())
          effects.emplace_back(effect.getEffect(), &operand);
      } else {
        effects.emplace_back(effect.getEffect(), &getInputsMutable()[argIdx]);
      }
    } else {
      // The trailing chain block arg is not a memory resource. Buffers are
      // captured operands, so their effects are already keyed on the captured
      // SSA values.
      effects.push_back(effect);
    }
  }
}

LogicalResult ParallelOp::verify() {
  auto inputs = getInputs();
  SmallVector<DeviceRefAttr> launchDevices = getLaunchDeviceList();

  // Launch count comes from launchDevices (the source of truth).  When
  // input bundles are present they must agree on width.
  size_t numLaunches = launchDevices.size();
  if (numLaunches == 0)
    return emitOpError("launchDevices must have at least one entry");

  // Each input bundle's elements must match launchDevices: same length and
  // same per-launch device.
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    auto bType = cast<BundleType>(input.getType());
    if (bType.getNumElements() != numLaunches)
      return emitOpError("bundle input #")
             << idx << " has " << bType.getNumElements()
             << " elements, expected " << numLaunches;
    for (auto [i, elemType] : llvm::enumerate(bType.getElementTypes())) {
      auto tensorTy = dyn_cast<TensorType>(elemType);
      if (!tensorTy)
        continue;
      DeviceRefAttr bundleDevice = tensorTy.getDeviceRef();
      if (bundleDevice != launchDevices[i])
        return emitOpError("bundle input #")
               << idx << " element #" << i << "'s device " << bundleDevice
               << " does not match launchDevices[" << i
               << "] = " << launchDevices[i];
    }
  }

  if (hasBuffers() != hasChain())
    return emitOpError("buffers and chain must be both present or both absent");

  if (hasBuffers()) {
    if (getBuffers().size() != numLaunches)
      return emitOpError("buffers count (")
             << getBuffers().size() << ") must match number of launches ("
             << numLaunches << ")";
  }

  Block &body = getBodyRegion().front();
  size_t expectedArgs = inputs.size() + (hasChain() ? 1 : 0);
  if (body.getNumArguments() != expectedArgs)
    return emitOpError("body must have ")
           << expectedArgs << " block argument(s) (one per bundle input"
           << (hasChain() ? " plus a trailing chain arg" : "") << "), got "
           << body.getNumArguments();

  // Chain threading contract: a chained parallel takes its in-chain through a
  // trailing !mo.chain block argument and yields the threaded out-chain as the
  // terminator's last operand. A chainless parallel does neither.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (hasChain()) {
    if (!isa<ChainType>(body.getArguments().back().getType()))
      return emitOpError(
          "trailing body block argument must be !mo.chain when the parallel "
          "is chained");
    if (yield.getNumOperands() == 0 ||
        !isa<ChainType>(yield.getOperands().back().getType()))
      return emitOpError(
          "mo.yield must produce a trailing !mo.chain when the parallel is "
          "chained");
  } else {
    for (Value operand : yield.getOperands())
      if (isa<ChainType>(operand.getType()))
        return emitOpError("chainless parallel must not yield a chain");
  }

  // GEX-3701 restricts the body to kernels (or ops that will become kernels),
  // `mo.bundled.expand`, and a small whitelist of view/chain ops.
  for (Operation &op : body.without_terminator()) {
    if (failed(verifyParallelBodyOp(op, *this)))
      return failure();
  }

  // Ensure that the types within each input bundle match the type of the
  // corresponding block argument.
  for (auto [group, bundleInput] : llvm::enumerate(inputs)) {
    Type blockArgType = body.getArgument(group).getType();
    auto bundleType = cast<BundleType>(bundleInput.getType());
    for (auto [launch, elemType] :
         llvm::enumerate(bundleType.getElementTypes())) {
      if (!typesMatchIgnoringDeviceId(elemType, blockArgType))
        return emitOpError("bundle input #")
               << group << " element #" << launch << " type " << elemType
               << " does not match block argument type " << blockArgType;
    }
  }

  auto yieldOp = cast<YieldOp>(body.getTerminator());
  size_t numYields = yieldOp.getNumOperands();
  // The yielded chain (when chained) maps to the trailing chain result, not a
  // bundle result.
  size_t numBundleYields = numYields - (hasChain() ? 1 : 0);
  if (numBundleYields < 1)
    return emitOpError("yield must produce at least one bundle result");

  size_t expectedResults = numBundleYields + (hasChain() ? 1 : 0);
  if (getNumResults() != expectedResults)
    return emitOpError("expected ")
           << expectedResults << " results, got " << getNumResults();

  for (size_t yieldIdx = 0; yieldIdx < numBundleYields; ++yieldIdx) {
    auto resultType = dyn_cast<BundleType>(getResult(yieldIdx).getType());
    if (!resultType)
      return emitOpError("result #")
             << yieldIdx << " must be an mo.bundle type";
    if (resultType.getNumElements() != numLaunches)
      return emitOpError("result bundle #")
             << yieldIdx << " has " << resultType.getNumElements()
             << " elements, expected " << numLaunches;
    for (auto [i, elemType] : llvm::enumerate(resultType.getElementTypes())) {
      auto tensorTy = dyn_cast<TensorType>(elemType);
      if (!tensorTy)
        continue;
      DeviceRefAttr bundleDevice = tensorTy.getDeviceRef();
      if (bundleDevice != launchDevices[i])
        return emitOpError("result bundle #")
               << yieldIdx << " element #" << i << "'s device " << bundleDevice
               << " does not match launchDevices[" << i
               << "] = " << launchDevices[i];
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> SequenceOp::getInputParams() { return {}; }
SmallVector<ParamDeclAttr> SequenceOp::getAllParams() { return {}; }

void SequenceOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, TypeRange resultTypes,
                       uint64_t streamId) {
  state.addOperands(inputs);
  state.addTypes(resultTypes);
  state.addAttribute(
      SequenceOp::getStreamIdAttrName(state.name),
      builder.getIntegerAttr(builder.getIntegerType(64, /*isSigned=*/false),
                             streamId));
  // One block argument per input, typed 1:1.
  Block &block = state.addRegion()->emplaceBlock();
  SmallVector<Type> argTypes(inputs.getTypes());
  SmallVector<Location> locs(argTypes.size(), state.location);
  block.addArguments(argTypes, locs);
}

LogicalResult SequenceOp::verify() {
  Block &body = getBodyRegion().front();

  // Inputs map 1:1 to block arguments (plain variadic list, no bundles).
  if (body.getNumArguments() != getInputs().size())
    return emitOpError("body must have ")
           << getInputs().size() << " block argument(s) (one per input), got "
           << body.getNumArguments();
  for (auto [idx, input] : llvm::enumerate(getInputs())) {
    if (body.getArgument(idx).getType() != input.getType())
      return emitOpError("input #") << idx << " type " << input.getType()
                                    << " does not match block argument type "
                                    << body.getArgument(idx).getType();
  }

  // Yield operands map 1:1 to results.
  auto yieldOp = cast<YieldOp>(body.getTerminator());
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("yield has ")
           << yieldOp.getNumOperands() << " operand(s) but op has "
           << getNumResults() << " result(s)";
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (yielded.getType() != getResult(idx).getType())
      return emitOpError("yield operand #")
             << idx << " type " << yielded.getType()
             << " does not match result type " << getResult(idx).getType();
  }

  // Nested `mo.sequence` is unsupported: the side-stream device-context
  // override installed during MOGG->MGP lowering is single-level per device.
  if (getOperation()->getParentOfType<MO::SequenceOp>())
    return emitOpError("nested `mo.sequence` is unsupported");

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastShapeOp::verify() {
  TensorType inputXType = getInputX().getType();
  TensorType inputYType = getInputY().getType();
  TensorType resultType = getResult().getType();
  // For dynamically ranked/shaped result, we defer to shape inference
  if (!resultType.hasStaticShape())
    return success();

  bool inputXIsStatic = inputXType.hasStaticShape();
  bool inputYIsStatic = inputYType.hasStaticShape();
  // Note that inputs and output are rank-1 Shape tensors
  int64_t inputXRank = inputXIsStatic ? inputXType.getIntDim(0) : 0;
  int64_t inputYRank = inputYIsStatic ? inputYType.getIntDim(0) : 0;
  int64_t resultRank = resultType.getIntDim(0);

  if (inputXIsStatic && inputXRank > resultRank) {
    return emitError("result rank must be no less than first input rank if "
                     "both are statically ranked");
  }

  if (inputYIsStatic && inputYRank > resultRank) {
    return emitError("result rank must be no less than second input rank if "
                     "both are statically ranked");
  }

  if (inputXIsStatic && inputYIsStatic &&
      resultRank != std::max(inputXRank, inputYRank)) {
    return emitError("output rank must be the max between 2 input ranks, if "
                     "all are statically ranked");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

/// Check if the input type can be broadcasted to the result type, assuming both
/// have static ranks.
static ErrorOrSuccess verifyBroadcastStaticRanks(TensorType inputTy,
                                                 TensorType resTy) {
  if (inputTy.hasRank() && resTy.hasRank()) {
    if (inputTy.getRank() > resTy.getRank())
      return Error("result rank must be at least the input rank");

    ArrayRef<TypedAttr> fromDims = inputTy.getShape().getValues();
    ArrayRef<TypedAttr> toDims = resTy.getShape().getValues();
    for (auto [fromDim, toDim] :
         llvm::zip(llvm::reverse(fromDims), llvm::reverse(toDims))) {
      auto fromIntDim = tryGetConstantInt(fromDim);
      auto toIntDim = tryGetConstantInt(toDim);
      if (fromIntDim && toIntDim) {
        // We can only assert anything if both dimensions are known constants.
        int64_t inDim = *fromIntDim;
        if (inDim > 1 && (inDim != *toIntDim))
          return Error("input shape not broadcastable to result shape");
      }
    }
  }
  return success();
}

LogicalResult BroadcastToOp::verify() {
  TensorType resTy = getType();
  TensorType newShapeTy = getNewShape().getType();

  if (!newShapeTy.hasStaticShape()) {
    if (!resTy.hasRank())
      return success();
    return emitError(
        "result must be dynamically ranked if new shape has unknown size");
  }

  if (!resTy.hasRank()) {
    return emitError(
        "result must be statically ranked if new shape has static size");
  }

  if ((int64_t)resTy.getRank() != newShapeTy.getIntDim(0))
    return emitError("new shape size and output rank must match");

  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), resTy))
    return emitError(err.getError());

  return success();
}

void BroadcastToOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TensorValue input, TensorValue broadcastLike,
                          TensorValue broadcastShape) {
  TensorType resultTy =
      input.getType().cloneWith(broadcastLike.getType().getShape());

  // Note broadcastShape may be null
  if (!broadcastShape)
    broadcastShape =
        MO::ShapeOfOp::create(odsBuilder, input.getLoc(), broadcastLike);

  build(odsBuilder, odsState, resultTy, input, broadcastShape);
}

//===----------------------------------------------------------------------===//
// Static::BroadcastToOp
//===----------------------------------------------------------------------===//

LogicalResult Static::BroadcastToOp::verify() {
  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), getType()))
    return emitError(err.getError());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto valType = cast<ShapedType>(getValue().getType()).getElementType();
  auto resType = getResult().getType().getElementType();
  // TODO(#28523): Restrict the value type of booleans to i8.
  if (resType.isInteger(1)) {
    if (!(valType.isInteger(1) || valType.isInteger(8) ||
          valType.isInteger(32) || valType.isInteger(64))) {
      return emitOpError("for boolean tensors, the element type of the value "
                         "must be an integer type, but got ")
             << valType;
    }
    return success();
  }
  if (valType != resType) {
    return emitOpError("element type of the value (")
           << valType << ") must match the element type of the result ("
           << resType << ")";
  }
  return success();
}

// Create an ElementsAttr from a list of primitive values.
namespace {
template <typename T>
ElementsAttr createElementsAttr(OpBuilder &odsBuilder, ArrayRef<T> values,
                                Type elementType) {
  if constexpr (std::is_same_v<T, bool>) {
    // For bool, we convert to i8 type.
    auto rankedAttrTy =
        RankedTensorType::get(values.size(), odsBuilder.getI8Type());
    auto apIntVals = llvm::map_to_vector(
        values, [](bool val) { return APInt(8, val, /*isSigned=*/false); });
    auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, apIntVals);
    return cast<ElementsAttr>(valueAttr);
  } else if constexpr (std::is_integral_v<T>) {
    return cast<ElementsAttr>(odsBuilder.getAttr<IntArrayElementsAttr>(
        values, mlir::IntegerType::Signed));
  } else if constexpr (std::is_floating_point_v<T>) {
    auto apFloatValues =
        llvm::map_to_vector(values, [](T value) { return APFloat(value); });
    return cast<ElementsAttr>(
        FloatArrayElementsAttr::get(apFloatValues, elementType));
  } else if constexpr (std::is_same_v<T, APInt>) {
    auto rankedAttrTy = RankedTensorType::get(values.size(), elementType);
    auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, values);
    return cast<ElementsAttr>(valueAttr);
  } else if constexpr (std::is_same_v<T, APFloat>) {
    return cast<ElementsAttr>(FloatArrayElementsAttr::get(values, elementType));
  }
}

template <typename T>
void buildConstant(OpBuilder &odsBuilder, OperationState &odsState,
                   ArrayRef<T> values, Type elementType,
                   DeviceRefAttr deviceRef) {
  if (!deviceRef)
    deviceRef = getHostDevice(odsBuilder.getContext());
  auto type =
      MO::TensorType::get({(int64_t)values.size()}, elementType, deviceRef);
  ElementsAttr valueAttr = createElementsAttr(odsBuilder, values, elementType);
  ConstantOp::build(odsBuilder, odsState, type, valueAttr);
}

template <typename T>
void buildConstantScalar(OpBuilder &odsBuilder, OperationState &odsState,
                         T value, Type elementType, DeviceRefAttr deviceRef) {
  if (!deviceRef)
    deviceRef = getHostDevice(odsBuilder.getContext());
  auto type =
      MO::TensorType::get(SmallVector<int64_t>{}, elementType, deviceRef);
  ElementsAttr valueAttr = createElementsAttr(
      odsBuilder, ArrayRef<T>(SmallVector<T>{value}), elementType);
  ConstantOp::build(odsBuilder, odsState, type, valueAttr);
}
} // namespace

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Type resultType, ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int64_t> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values,
                odsBuilder.getIntegerType(64, /*isSigned=*/true), deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int32_t> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values,
                odsBuilder.getIntegerType(32, /*isSigned=*/true), deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APInt> values, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isInteger());
  buildConstant(odsBuilder, odsState, values, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APFloat> values, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isIntOrFloat() && !elementType.isInteger());
  buildConstant(odsBuilder, odsState, values, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<bool> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values, odsBuilder.getI1Type(),
                deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value,
                      odsBuilder.getIntegerType(64, /*isSigned=*/true),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int32_t value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value,
                      odsBuilder.getIntegerType(32, /*isSigned=*/true),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APFloat value, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isIntOrFloat() && !elementType.isInteger());
  buildConstantScalar(odsBuilder, odsState, value, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APInt value, Type elementType, DeviceRefAttr deviceRef) {
  assert(elementType.isInteger());
  buildConstant(odsBuilder, odsState, ArrayRef<APInt>(value), elementType,
                deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       float value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getF32Type(),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       double value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getF64Type(),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       bool value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getI1Type(),
                      deviceRef);
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOp::verify() {
  auto scalarDType = cast<MO::ScalarType>(getType()).getDType();
  if (auto value = dyn_cast<IntegerAttr>(getValue())) {
    if (!isa<IntegerType>(scalarDType))
      return emitError(
          "mo.constant.scalar expected the dtype to be an integer");
    if (value.getType().getIntOrFloatBitWidth() !=
        scalarDType.getIntOrFloatBitWidth())
      return emitError("mo.constant.scalar expected bitwidth of the dtype and "
                       "attribute to match");
  } else if (auto value = dyn_cast<FloatAttr>(getValue())) {
    if (!isa<FloatType>(scalarDType))
      return emitError("mo.constant.scalar expected the dtype to be a float");
    if (value.getType().getIntOrFloatBitWidth() !=
        scalarDType.getIntOrFloatBitWidth())
      return emitError("mo.constant.scalar expected bitwidth of the dtype and "
                       "attribute to match");
  }
  return success();
}

void ConstantScalarOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             TypedAttr value) {
  auto type = value.getType();

  DType dtype;

  if (auto integerType = dyn_cast<IntegerType>(type))
    dtype = M::getEquivalentDType(integerType);
  else
    dtype = M::getEquivalentDType(cast<FloatType>(type));

  auto resultType = odsBuilder.getType<ScalarType>(dtype);
  build(odsBuilder, odsState, resultType, value);
}

//===----------------------------------------------------------------------===//
// BufferCreateOp
//===----------------------------------------------------------------------===//

void BufferCreateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           Type result, Attribute initValue) {
  if (initValue)
    odsState.addAttribute(getInitValueAttrName(odsState.name), initValue);
  odsState.addTypes(result);
}

LogicalResult BufferCreateOp::verify() {
  Attribute initValue = getInitValueAttr();
  if (!initValue)
    return success();

  Type bufferElementType =
      castFromTensorLike(getResult().getType()).getElementType();

  // The init value is a single-element elements attribute (the form the Python
  // API emits) or a scalar typed attribute.
  Type valueElementType;
  if (auto elts = dyn_cast<ElementsAttr>(initValue)) {
    if (elts.getNumElements() != 1)
      return emitError(
          "mo.buffer.create initValue must be a single scalar element");
    valueElementType = elts.getElementType();
  } else if (auto typed = dyn_cast<TypedAttr>(initValue)) {
    valueElementType = typed.getType();
  } else {
    return emitError("mo.buffer.create initValue must be a typed scalar or "
                     "elements attribute");
  }

  if (isa<FloatType>(bufferElementType) != isa<FloatType>(valueElementType) ||
      bufferElementType.getIntOrFloatBitWidth() !=
          valueElementType.getIntOrFloatBitWidth())
    return emitError("mo.buffer.create initValue element type must match the "
                     "buffer's element type");
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs < 1)
    return emitError("concat must have at least 1 input tensor");

  SmallVector<TensorValue> tensors;
  for (Value input : getInputs())
    tensors.emplace_back(cast<TensorValue>(input));

  tensors.emplace_back(getResult());
  if (failed(::Impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  auto elementTypes = llvm::map_range(
      tensors, [](TensorValue val) { return val.getType().getElementType(); });

  if (!llvm::all_equal(elementTypes)) {
    return emitError("input and output element types must match");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

/// Verify that the sum of `outputs` shapes along `splitAxis` matches the input
/// shape, and that the other dimensions match.
static ErrorOrSuccess verifyMatchingSplitShapes(ArrayRef<TensorValue> outputs,
                                                TensorType inputTy,
                                                int64_t splitAxis) {
  int64_t splitDimSum = 0;
  for (auto res : outputs) {
    TensorType resTy = res.getType();
    for (auto [i, dimSize] : llvm::enumerate(resTy.getIntDims())) {
      if ((int64_t)i == splitAxis) {
        splitDimSum += dimSize;
      } else if (dimSize != inputTy.getIntDim(i)) {
        return Error(
            "input and output static shapes must match except split axis");
      }
    }
  }

  if (splitDimSum != inputTy.getIntDim(splitAxis)) {
    return Error(
        "sum of output shapes along split axis must match input shape");
  }

  return success();
}

LogicalResult SplitOp::verify() {
  SmallVector<TensorValue> tensors;
  tensors.emplace_back(getInput());
  for (Value output : getResults())
    tensors.emplace_back(cast<TensorValue>(output));

  if (failed(::Impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  TensorType inputTy = getInput().getType();
  TensorType resultTy = cast<TensorType>(getResults().front().getType());
  if (inputTy.getDtype() != resultTy.getDtype())
    return emitError("input and output element types must match");

  // `axis` is a compile-time `index` attribute.
  if (inputTy.hasStaticShape()) {
    int64_t axis = getAxisAttr().getInt();
    int64_t inputRank = inputTy.getRank();
    int64_t normalizedAxis = normalizeAxis(axis, inputRank);
    if ((normalizedAxis < 0) || (inputRank <= normalizedAxis))
      return emitError("split axis must be in range [-rank, rank)");

    if (resultTy.hasStaticShape()) {
      ErrorOrSuccess result = verifyMatchingSplitShapes(
          ArrayRef<TensorValue>(tensors).drop_front(), inputTy, normalizedAxis);
      if (result.isError())
        return emitError(result.getError());
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  TensorType paramsType = getInput().getType();
  TensorType indicesType = getIndices().getType();
  TensorType resultType = getResult().getType();

  if (paramsType.getDtype() != resultType.getDtype())
    return emitError("input and output must have the same element type");

  // Nothing we can say if either input or indices is dynamically ranked
  if (!paramsType.hasRank() || !indicesType.hasRank())
    return success();

  if (!resultType.hasRank()) {
    return emitError(
        "result must be ranked if both param and indices are ranked");
  }

  if (paramsType.getRank() - 1 + indicesType.getRank() !=
      resultType.getRank()) {
    return emitError("result rank must be 1 less than the sum of param and "
                     "indices ranks if they are all static");
  }

  return success();
}

LogicalResult GatherSumOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType indicesType = getIndices().getType();
  TensorType resultType = getResult().getType();

  // Operand constraints (rank-2 table, rank-2 si32 indices) are enforced
  // by the TableGen confined operand types; this verifier pins only the
  // result shape: the reduce.add output [batch, 1, ...] with the
  // multi-hot axis kept at size 1 (or the rank-2 equivalent).
  if (resultType.getRank() != 2 && resultType.getRank() != 3)
    return emitError("gather_sum result must be rank 2 or 3");

  // Pin the result to the kernel's derived shape: batch from the indices,
  // multi-hot axis kept at 1, embedding dim from the table. A too-small
  // declared result would otherwise write out of bounds, and unlike most
  // ops this one has no shape inference that would catch the mismatch.
  // TypedAttr equality covers both constant dims and matching parametric
  // dims (e.g. result and table both `N`); a constant-vs-parametric pair
  // is left for shape materialization to decide.
  bool batchOk = resultType.getDim(0) == indicesType.getDim(0);
  bool midAxisKept = resultType.getRank() == 2 ||
                     MOSH::ShapeAttr::getStaticDim(resultType.getDim(1)) == 1;
  auto embDim =
      resultType.getRank() == 2 ? resultType.getDim(1) : resultType.getDim(2);
  if (!batchOk || !midAxisKept || embDim != inputType.getDim(1))
    return emitError("gather_sum result must be [batch, 1, emb_dim] "
                     "(or [batch, emb_dim]) matching the indices batch "
                     "and table width");

  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  // `axis` is a compile-time `index` attribute; its range is validated during
  // shape materialization.
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();

  if (inputTy.hasStaticShape() && resTy.hasStaticShape() &&
      inputTy.getShape() != resTy.getShape()) {
    return emitError("input and result shapes must match if both static");
  }

  // Input, updates, indices and result must share a rank when ranked.
  uint64_t expectedRank = 0;
  bool haveExpected = false;
  auto rankMatches = [&](TensorValue tensor) -> bool {
    TensorType ty = tensor.getType();
    if (!ty.hasRank())
      return true;
    if (!haveExpected) {
      expectedRank = ty.getRank();
      haveExpected = true;
      return true;
    }
    return ty.getRank() == expectedRank;
  };
  if (!rankMatches(getInput()) || !rankMatches(getUpdates()) ||
      !rankMatches(getIndices()) || !rankMatches(getResult())) {
    return emitError(
        "output and inputs (except axis) of a scatter op must have the "
        "same rank");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  DType xDType = getX().getType().getDtype();
  DType yDType = getY().getType().getDtype();
  DType resultDType = getResult().getType().getDtype();

  if (xDType != yDType)
    return emitError("operands must have the same element type");
  if (xDType != resultDType)
    return emitError(
        "operands and result type must have the same element type");

  return success();
}

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

LogicalResult ArgNonzeroOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();

  if (!resTy.hasRank() || resTy.getRank() != 2)
    return emitError("mo.arg_nonzero: invalid output shape: rank must be 2");

  if ((inputTy.hasRank() && (int64_t)inputTy.getRank() > 0 &&
       resTy.getIntDim(1) != (int64_t)inputTy.getRank()) ||
      (!inputTy.hasRank() && !ShapedType::isDynamic(resTy.getIntDim(1))))
    return emitError(
        "mo.arg_nonzero: invalid output shape: inner_dim != input_rank");

  return success();
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult
verifyWithRankedInput(Op *op, int64_t axis, int64_t resDim, TensorType inputTy,
                      size_t &dynamicOutDims, size_t &reducedDims) {
  auto inDim = inputTy.getShape().getIntDim(axis);
  if (mlir::ShapedType::isDynamic(resDim) ||
      mlir::ShapedType::isDynamic(inDim)) {
    ++dynamicOutDims;
    return success();
  }
  if (inDim != resDim) {
    if (resDim != 1)
      return op->emitOpError(
          "mo.reduce.arg_max: output dimensions must match input "
          "dimensions, except at the reduction axis where the "
          "output dimension must be 1");
    ++reducedDims;
  }
  return success();
}

static void verifyWithUnrankedInput(int64_t axis, int64_t resDim,
                                    size_t &dynamicOutDims,
                                    size_t &reducedDims) {
  if (mlir::ShapedType::isDynamic(resDim)) {
    ++dynamicOutDims;
    return;
  }
  if (resDim == 1)
    ++reducedDims;
}

template <typename Op>
static LogicalResult argVerify(Op *op) {
  static_assert(std::is_same_v<Op, ArgMaxOp> || std::is_same_v<Op, ArgMinOp>,
                "input op must be either an ArgMax or ArgMin op");

  TensorType resTy = op->getResult().getType();
  TensorType inputTy = op->getInput().getType();

  // TODO(#12939): Refactor this into verifyReduction.

  if ((inputTy.hasRank() && resTy.hasRank()) &&
      (inputTy.getRank() != resTy.getRank()))
    return op->emitOpError(Op::getOperationName() +
                           ": output rank must match input rank");

  if (resTy.hasRank()) {
    // `axis` is a compile-time `index` attribute and is always known.
    int64_t optAxis = op->getAxisAttr().getInt();
    size_t reducedDims = 0;
    size_t dynamicOutDims = 0;

    for (int64_t axis = 0, rank = resTy.getRank(); axis < rank; ++axis) {
      int64_t resDim = resTy.getShape().getIntDim(axis);

      if ((axis == optAxis) && (resDim != 1) &&
          !mlir::ShapedType::isDynamic(resDim))
        return op->emitOpError(
            Op::getOperationName() +
            ": output dimensions must be 1 at the reduction axis");

      // Note: we don't check that non-reduction axes match directly. The code
      // below does it, which handles the dynamic cases as well.

      if (inputTy.hasRank()) {
        if (failed(verifyWithRankedInput(op, axis, resDim, inputTy,
                                         dynamicOutDims, reducedDims)))
          return failure();
      } else {
        verifyWithUnrankedInput(axis, resDim, dynamicOutDims, reducedDims);
      }
    }

    if ((reducedDims > 1) || (!reducedDims && !dynamicOutDims)) {
      return op->emitOpError(
          Op::getOperationName() +
          ": output dimensions must match input "
          "dimensions, except for exactly one dimension, the reduction axis");
    }
  }

  return success();
}

template <typename TOp>
void buildReductionOp(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, int64_t axis, TensorType outputTy) {
  TOp::build(odsBuilder, odsState, outputTy, input,
             odsBuilder.getIndexAttr(axis));
}

LogicalResult ArgMaxOp::verify() { return argVerify(this); }

void ArgMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ArgMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// ArgMinOp
//===----------------------------------------------------------------------===//

LogicalResult ArgMinOp::verify() { return argVerify(this); }

void ArgMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ArgMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

LogicalResult CumSumOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getDtype() != resultType.getDtype())
    return emitOpError("Expect input and result DTypes to be the same.");

  auto inDimAttrs = inputType.getShape();
  auto outDimAttrs = resultType.getShape();
  if (inDimAttrs && outDimAttrs) {
    for (auto [inDimAttr, outDimAttr] :
         llvm::zip(inDimAttrs.getValues(), outDimAttrs.getValues())) {
      auto inDimInt = tryGetConstantInt(inDimAttr);
      auto outDimInt = tryGetConstantInt(outDimAttr);
      if (inDimInt && outDimInt && *inDimInt != *outDimInt)
        return emitError("statically known dimensions must match");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Windowed ops
//===----------------------------------------------------------------------===//

LogicalResult AvgPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult AvgPoolCeilOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult MaxPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult MaxPoolCeilOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

LogicalResult ConvOp::verify() {
  // The sliding window interface already checks input/result ranks; this
  // gives a more specific error.
  if (failed(
          ::Impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes({getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult ConvTransposeOp::verify() {
  // The sliding window interface already checks input/result ranks; this
  // gives a more specific error.
  if (failed(
          ::Impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes(
      {getDilations(), getStrides(), getOutputPaddings()});
}

//===----------------------------------------------------------------------===//
// Matmul definitions.
//===----------------------------------------------------------------------===//

void BatchMatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type resultTy, Value input_a, Value input_b,
                          ArrayRef<KGEN::ParamDeclAttr> decls) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({resultTy});
  if (!decls.empty()) {
    odsState.addAttribute(
        getOutputParamDeclsAttrName(odsState.name),
        KGEN::ParamDeclArrayAttr::get(odsBuilder.getContext(), decls));
  }
}

LogicalResult BatchMatmulOp::verify() {
  TensorType aType = getTensorA().getType();
  TensorType bType = getTensorB().getType();
  TensorType cType = getTensorResult().getType();

  // Note MO_MatmulLike interface verifies ranks are consistent between ops.
  auto violatedRankProperty = [&](TensorType ty) -> bool {
    return ty.hasRank() && ty.getRank() < 3;
  };
  if (violatedRankProperty(aType) || violatedRankProperty(bType) ||
      violatedRankProperty(cType)) {
    return emitError("All operands and results must be of at least rank 3.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

LogicalResult RangeOp::verify() {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = getOutputParamDecls();
  if (!paramDecls.empty()) {
    if (paramDecls.size() != 1)
      return emitError("cannot declare more than one parameter");
    if (!ShapeAttr::isDimType(paramDecls[0].getType()))
      return emitError("can only declare dimension parameters");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

LogicalResult RebindOp::verify() {
  TensorType inTy = getInput().getType();
  TensorType outTy = getType();
  // Disallow rebind to a different device.
  if (inTy.getDeviceRef() && outTy.getDeviceRef() &&
      (inTy.getDeviceRef() != outTy.getDeviceRef()))
    return emitError("input device " + inTy.getDeviceRef().toString() +
                     " must match result device " +
                     outTy.getDeviceRef().toString() + " in rebind");

  // Nothing to check if either type is dynamically ranked
  if (!inTy.hasRank() || !outTy.hasRank())
    return success();

  if (inTy.getRank() != outTy.getRank())
    return emitError("input and output static ranks must match");

  SmallDenseMap<KGEN::ParamDeclRefAttr, TypedAttr> dimParamToConst;
  auto tryUnifyParamToConstDim = [&](TypedAttr srcDimAttr,
                                     TypedAttr dstDimAttr) -> LogicalResult {
    auto srcDimParamRef = dyn_cast<KGEN::ParamDeclRefAttr>(srcDimAttr);
    if (srcDimParamRef && tryGetConstantInt(dstDimAttr)) {
      auto [cachedEntry, _] =
          dimParamToConst.try_emplace(srcDimParamRef, dstDimAttr);
      if (cachedEntry->second != dstDimAttr)
        return failure();
    }
    return success();
  };
  auto inDimAttrs = inTy.getShape();
  auto outDimAttrs = outTy.getShape();
  if (inDimAttrs && outDimAttrs) {
    for (auto [inDimAttr, outDimAttr] :
         llvm::zip(inDimAttrs.getValues(), outDimAttrs.getValues())) {
      auto inDimInt = tryGetConstantInt(inDimAttr);
      auto outDimInt = tryGetConstantInt(outDimAttr);
      if (inDimInt && outDimInt && *inDimInt != *outDimInt)
        return emitError("statically known dimensions must match");
      else if (failed(tryUnifyParamToConstDim(inDimAttr, outDimAttr)) ||
               failed(tryUnifyParamToConstDim(outDimAttr, inDimAttr)))
        return emitError("one dimension parameter cannot bind to different "
                         "static dimensions");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceMulOp
//===----------------------------------------------------------------------===//

void ReduceMulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMulOp>(odsBuilder, odsState, input, axis, outputTy);
}

// `axis`, made non-negative.
static std::optional<int64_t> normalizedReductionAxis(TensorValue input,
                                                      int64_t axis) {
  if (axis < 0)
    axis += input.getType().getRank();
  return axis;
}

std::optional<int64_t> ReduceMulOp::getReductionAxis() {
  return normalizedReductionAxis(getInput(), getAxisAttr().getInt());
}

std::optional<int64_t> ArgMaxOp::getReductionAxis() {
  return normalizedReductionAxis(getInput(), getAxisAttr().getInt());
}

std::optional<int64_t> ArgMinOp::getReductionAxis() {
  return normalizedReductionAxis(getInput(), getAxisAttr().getInt());
}

//===----------------------------------------------------------------------===//
// ReduceAddOp, ReduceMinOp, ReduceMaxOp
//===----------------------------------------------------------------------===//

static void buildImplicitAxisReduction(OpBuilder &builder,
                                       OperationState &state, TensorValue input,
                                       int64_t axis) {
  TensorType inputTy = input.getType();
  state.addOperands(input);
  state.addTypes(
      inputTy.cloneWithNewDim(axis < 0 ? axis + inputTy.getRank() : axis, 1));
}

static std::optional<int64_t> implicitReductionAxis(TensorValue input,
                                                    TensorValue result) {
  ErrorOr<int64_t> axis = getReductionAxis(input, result.getType());
  if (axis.isError())
    return std::nullopt;
  return *axis;
}

// A reduction that changes no dimension returns its input.
static OpFoldResult foldImplicitAxisReduction(TensorValue input,
                                              TensorValue result) {
  if (input.getType() == result.getType())
    return input;
  return {};
}

#define DEFINE_IMPLICIT_AXIS_REDUCTION(OpName)                                 \
  void OpName::build(OpBuilder &builder, OperationState &state,                \
                     TensorValue input, int64_t axis) {                        \
    buildImplicitAxisReduction(builder, state, input, axis);                   \
  }                                                                            \
  std::optional<int64_t> OpName::getReductionAxis() {                          \
    return implicitReductionAxis(getInput(), getResult());                     \
  }                                                                            \
  OpFoldResult OpName::fold(FoldAdaptor) {                                     \
    return foldImplicitAxisReduction(getInput(), getResult());                 \
  }

DEFINE_IMPLICIT_AXIS_REDUCTION(ReduceAddOp)
DEFINE_IMPLICIT_AXIS_REDUCTION(ReduceMinOp)
DEFINE_IMPLICIT_AXIS_REDUCTION(ReduceMaxOp)
#undef DEFINE_IMPLICIT_AXIS_REDUCTION

//===----------------------------------------------------------------------===//
// TopK / BottomK
//===----------------------------------------------------------------------===//

template <typename OrderKOpT>
LogicalResult verifyOrderedKOp(OrderKOpT op) {
  auto output1Shape = op.getValues().getType().getShape();
  auto output2Shape = op.getIndices().getType().getShape();
  if (output1Shape != output2Shape) {
    return op.emitOpError(
               "values and indices should have the same shape but got: ")
           << output1Shape << " vs " << output2Shape << "\n";
  }
  return success();
}

LogicalResult TopKOp::verify() { return verifyOrderedKOp(*this); }

LogicalResult BottomKOp::verify() { return verifyOrderedKOp(*this); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

template <typename T>
inline LogicalResult checkReshapeInOutTypesCompatible(T op) {
  TensorType inputTy = op.getInput().getType();
  TensorType resultTy = op.getType();
  return checkReshapeShapesCompatible(inputTy.getShape(), resultTy.getShape(),
                                      [&]() { return op.emitOpError(); });
}

LogicalResult ReshapeOp::verify() {
  LogicalResult result = checkReshapeInOutTypesCompatible(*this);
  if (result.failed())
    return result;

  auto shapeInput =
      dyn_cast_if_present<ConstantOp>(getNewShape().getDefiningOp());
  if (!shapeInput)
    return success();

  ElementsAttr attr = shapeInput.getValue();
  auto dimIntegers =
      tryGetConstantBlob<int64_t>(attr, getNewShape().getType().getDtype());
  if (!dimIntegers.has_value())
    return success();

  int64_t requestedElements = 1;
  for (int64_t dim : dimIntegers.value()) {
    if (dim == -1)
      return emitOpError("-1 size unsupported in requested shape");
    requestedElements *= dim;
  }

  std::optional<int64_t> resultElements =
      getResult().getType().getNumElements();
  if (resultElements.has_value() &&
      resultElements.value() != requestedElements) {
    return emitOpError("Expected requested shape and output shape to have the "
                       "same number of elements requested ")
           << requestedElements << " but expected " << resultElements.value();
  }

  std::optional<int64_t> inputElements = getInput().getType().getNumElements();
  if (inputElements.has_value() && inputElements.value() != requestedElements) {
    return emitOpError("Expected requested shape and input shape to have the "
                       "same number of elements requested ")
           << requestedElements << " but expected " << inputElements.value();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnsqueezeShapeOp
//===----------------------------------------------------------------------===//
LogicalResult UnsqueezeShapeOp::verify() {
  if (getInputShape().getType().getDtype() != getResult().getType().getDtype())
    return emitOpError("Expect in and out DTypes to be the same type");

  std::optional<int64_t> inputShapeSize =
      getInputShape().getType().getNumElements();

  // Check indices don't point to the same index. Negative indices are
  // normalized first, so that -1 and 5 on a rank-6 output count as the same.
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(getPaddingIndices());
  if (maybeIndices) {
    SmallVector<int64_t> &indices = maybeIndices.value();
    if (inputShapeSize) {
      int64_t outputRank = *inputShapeSize + indices.size();
      for (int64_t &index : indices) {
        if (index < -outputRank || index >= outputRank)
          return emitOpError("Unsqueeze index not in [-out_rank, out_rank)");
        if (index < 0)
          index += outputRank;
      }
    }
    llvm::sort(indices);
    for (size_t i = 1; i < indices.size(); ++i) {
      if (indices[i - 1] == indices[i])
        return emitOpError("Padding indices pad the same dimension twice");
    }
  }

  std::optional<int64_t> paddingIndicesSize =
      getPaddingIndices().getType().getNumElements();
  std::optional<int64_t> outputSize = getResult().getType().getNumElements();
  if (inputShapeSize.has_value() && paddingIndicesSize.has_value() &&
      outputSize.has_value() &&
      inputShapeSize.value() + paddingIndicesSize.value() !=
          outputSize.value()) {
    return emitOpError("Output size is not the sum of the input sizes");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SqueezeShapeOp
//===----------------------------------------------------------------------===//

LogicalResult SqueezeShapeOp::verify() {
  if (getInputShape().getType().getDtype() != getResult().getType().getDtype())
    return emitOpError("Expect in and out DTypes to be the same type");

  std::optional<int64_t> inputShapeSize =
      getInputShape().getType().getNumElements();

  // Check indices don't point to the same index. Negative indices are
  // normalized first, so that -1 and 3 on a rank-4 input count as the same.
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(getRemoveIndices());
  if (maybeIndices) {
    SmallVector<int64_t> &indices = maybeIndices.value();
    if (inputShapeSize) {
      int64_t inputRank = *inputShapeSize;
      for (int64_t &index : indices) {
        if (index < -inputRank || index >= inputRank)
          return emitOpError("Squeeze index not in [-in_rank, in_rank)");
        if (index < 0)
          index += inputRank;
      }
    }
    llvm::sort(indices);
    for (size_t i = 1, e = indices.size(); i < e; ++i)
      if (indices[i - 1] == indices[i])
        return emitOpError("Padding indices pad the same dimension twice");
  }

  std::optional<int64_t> paddingIndicesSize =
      getRemoveIndices().getType().getNumElements();
  std::optional<int64_t> outputSize = getResult().getType().getNumElements();
  if (inputShapeSize && paddingIndicesSize && outputSize &&
      inputShapeSize.value() - paddingIndicesSize.value() !=
          outputSize.value()) {
    return emitOpError("Output size is not the difference of the input sizes");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Static::ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult Static::ReshapeOp::verify() {
  return checkReshapeInOutTypesCompatible(*this);
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeOfOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();
  if (resTy.hasStaticShape()) {
    if (!inputTy.hasRank())
      return emitError("input must have static rank if result has known size");
    if ((int64_t)inputTy.getRank() != resTy.getIntDim(0))
      return emitError("input rank and output size must match");
    return success();
  }

  if (inputTy.hasRank())
    return emitError("input must have dynamic rank if result has unknown size");
  return success();
}

void ShapeOfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, int64_t width) {
  ASSERT_STREAM(llvm::is_contained({32, 64}, width),
                << "`mo.shape_of` returns si32 or si64");
  Type intTy = odsBuilder.getIntegerType(width, /*isSigned=*/true);

  int64_t outputSize = mlir::ShapedType::kDynamic;
  if (input.getType().hasRank())
    outputSize = input.getType().getRank();

  // TODO(#18749): don't introduce `?` with operations.
  build(odsBuilder, odsState,
        MO::TensorType::get({outputSize}, intTy,
                            getHostDevice(odsBuilder.getContext())),
        input);
}

void ShapeOfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input) {
  build(odsBuilder, odsState, input, /*width=*/32);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType startTy = getStart().getType();
  TensorType stopTy = getStop().getType();
  TensorType stepTy = getStep().getType();
  if (inputTy.hasRank()) {
    int64_t inputRank = inputTy.getRank();
    if (startTy.hasStaticShape() && startTy.getIntDim(0) != inputRank)
      return emitError("start shape must be [inputRank] if it's static");

    if (stopTy.hasStaticShape() && stopTy.getIntDim(0) != inputRank)
      return emitError("stop shape must be [inputRank] if it's static");

    if (stepTy.hasStaticShape() && stepTy.getIntDim(0) != inputRank)
      return emitError("step shape must be [inputRank] if it's static");

  } else {
    if (startTy.hasStaticShape()) {
      return emitError(
          "start shape must be dynamic if input is has dynamic rank");
    }

    if (stopTy.hasStaticShape()) {
      return emitError(
          "stop shape must be dynamic if input is has dynamic rank");
    }

    if (stepTy.hasStaticShape()) {
      return emitError(
          "step shape must be dynamic if input is has dynamic rank");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  // The permutation has known shape iff the argument and result are
  // statically ranked.
  TensorType permTy = getPerm().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!permTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if perm "
                       "has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if perm "
                     "has known shape");
  }

  size_t opRank = resTy.getRank();
  if (inputTy.getRank() != opRank)
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that perm has rank = 1.
  if ((int64_t)opRank != permTy.getIntDim(0))
    return emitError("input and result ranks do not match perm size");

  // A valid perm must be a permutation of [0, ..., rank - 1].
  if (auto maybePermVec = tryGetConstantIntVector(getPerm())) {
    auto permVec = maybePermVec.value();
    auto identityPerm =
        llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
    SmallVector<int64_t> sortedPerm(permVec);
    llvm::sort(sortedPerm);
    if (!llvm::equal(sortedPerm, identityPerm))
      return emitError("invalid permutations");

    if (inputTy.hasStaticShape() && resTy.hasStaticShape() &&
        !llvm::equal(resTy.getShape().getValues(),
                     permute(inputTy.getShape().getValues(), permVec)))
      return emitError("output shape is invalid based on constant permutation");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

LogicalResult TileOp::verify() {
  // `repeats` has known shape iff the argument and result are statically
  // ranked.
  TensorType repeatsTy = getRepeats().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!repeatsTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if "
                       "repeats has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if "
                     "repeats has known shape");
  }

  if (inputTy.getRank() != resTy.getRank())
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that repeats has rank = 1.
  if ((int64_t)resTy.getRank() != repeatsTy.getIntDim(0))
    return emitError("input and result ranks do not match repeats size");
  return success();
}

//===----------------------------------------------------------------------===//
// Linear Algebra Ops
//===----------------------------------------------------------------------===//

LogicalResult LinAlg::BandPartOp::verify() {
  // TODO: Refactor this into a trait when we see another op that does this
  if (getInput().getType() != getResult().getType())
    return emitError("first operand and the result must have the same type");
  return success();
}

//===----------------------------------------------------------------------===//
// InvokeShapeFuncOp
//===----------------------------------------------------------------------===//

LogicalResult InvokeShapeFuncOp::verify() {
  if (llvm::any_of(getOutputParamDecls(), [&](ParamDeclAttr decl) {
        return isa<MOSH::ShapeType>(decl.getType());
      }))
    return emitError() << "dynamic rank isn't supported in shape function";
  return success();
}

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//

LogicalResult LayoutTransformOp::verify() {
  if (!getInput().getType().getLayoutAttr())
    return emitError() << "input type must contain layout annotation";

  if (!getResult().getType().getLayoutAttr())
    return emitError() << "output type must contain layout annotation";

  return success();
}

LogicalResult
GuardOp::inferReturnTypes(MLIRContext *context,
                          std::optional<Location> location, Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes = llvm::to_vector(TypeRange(adaptor.getInputs()));
  return success();
}

ParseResult ChainCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> types;

  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(types))
      return failure();

    if (operands.size() != types.size())
      return parser.emitError(parser.getNameLoc(),
                              "mismatched number of operands and types");

    if (parser.resolveOperands(operands, types, parser.getNameLoc(),
                               result.operands))
      return failure();
  }

  result.types.push_back(ChainType::get(parser.getContext()));

  return success();
}

void ChainCreateOp::print(OpAsmPrinter &p) {
  p << "(";
  p.printOperands(getOperands());
  p << ")";
  p.printOptionalAttrDict(getOperation()->getAttrs());
  if (!getOperands().empty()) {
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
  }
}

void AssertOp::build(OpBuilder &builder, OperationState &state, Value cond,
                     const Twine &message) {
  auto attr = StringAttr::get(message.str(),
                              KGEN::StringType::get(builder.getContext()));
  build(builder, state, {}, cond, cast<TypedAttr>(attr));
}

static TypedAttr emitStaticCallImpl(MLIRContext *ctx,
                                    ArrayRef<TypedAttr> operands,
                                    const Twine &message) {
  SmallVector<std::pair<StringAttr (*)(mlir::OperationName), Attribute>>
      msgAttrNameFuncAndArg = {
          {AssertOp::getMessageAttrName,
           StringAttr::get(message, KGEN::StringType::get(ctx))}};
  auto chainTy = ChainType::get(ctx);
  return KGEN::emitMLIROperationCall(AssertOp::getOperationName(),
                                     msgAttrNameFuncAndArg, operands, chainTy);
}

TypedAttr AssertOp::emitStaticCall(TypedAttr cond, const Twine &message) {
  return emitStaticCallImpl(cond.getContext(), {cond}, message);
}

static TypedAttr emitStaticCall(TypedAttr chain, TypedAttr cond,
                                const Twine &message) {
  return emitStaticCallImpl(cond.getContext(), {chain, cond}, message);
}

static void linkAssertPair(mlir::RewriterBase &rewriter,
                           Operation *prevAssertOp,
                           Operation *currentAssertOp) {

  MLIRContext *ctx = currentAssertOp->getContext();

  auto prevAssertDeclareOp = dyn_cast<KGEN::ParamDeclareOp>(prevAssertOp);
  auto currentAssertDeclareOp = dyn_cast<KGEN::ParamDeclareOp>(currentAssertOp);

  // Grab the names of some relevant attrs so we don't hard code them later on.
  auto paramDeclOpName =
      mlir::OperationName(KGEN::ParamDeclareOp::getOperationName(), ctx);
  auto assertOpName = mlir::OperationName(AssertOp::getOperationName(), ctx);

  auto prevChain = prevAssertDeclareOp.getValue();

  auto operatorAttr = cast<KGEN::ParamOperatorAttr>(prevChain);
  auto operands = operatorAttr.getOperands();

  auto mlirOp = cast<KGEN::MLIROpAttr>(operands[0]);
  auto dictAttr = mlirOp.getAttrs();
  auto message = dictAttr.get(AssertOp::getMessageAttrName(assertOpName));

  StringAttr messageAttr = cast<mlir::StringAttr>(message);
  auto errorMsg = messageAttr.strref();

  auto currentOperands =
      cast<KGEN::ParamOperatorAttr>(currentAssertDeclareOp.getValue())
          .getOperands()
          .drop_front();

  // The new assert consumes prevAssertOp's output chain.
  auto newAttr = emitStaticCall(
      KGEN::ParamDeclRefAttr::get(prevAssertDeclareOp.getParamDecl()),
      currentOperands[0], errorMsg);

  rewriter.startOpModification(currentAssertDeclareOp);
  currentAssertDeclareOp->setAttr(
      KGEN::ParamDeclareOp::getValueAttrName(paramDeclOpName), newAttr);
  rewriter.finalizeOpModification(currentAssertDeclareOp);
}

static void linkAssertSequence(mlir::RewriterBase &rewriter,
                               ArrayRef<KGEN::ParamDeclareOp> asserts) {

  ASSERT_STREAM(!asserts.empty(),
                << "linkAssertSequence received an empty sequence of asserts!");

  KGEN::ParamDeclareOp prevAssert = asserts.front();

  for (auto assert : asserts.drop_front()) {
    linkAssertPair(rewriter, prevAssert, assert);
    prevAssert = assert;
  }
}

void AssertOp::materializeAllAssertLogic(
    mlir::RewriterBase &rewriter, Operation *op,
    ArrayRef<KGEN::ParamDeclareOp> asserts) {

  if (asserts.empty())
    return;

  linkAssertSequence(rewriter, asserts);

  mlir::RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Materialize the KGEN param-level chains as SSA values.
  SmallVector<Value> allChains;
  for (auto assert : asserts) {
    auto declRefAttr = KGEN::ParamDeclRefAttr::get(
        cast<KGEN::ParamDeclAttr>(assert.getParamDecl()));

    allChains.push_back(
        MOSH::ParamToValueOp::create(rewriter, op->getLoc(), declRefAttr));
  }

  // fan in all the assertion chains as the guard op takes a single chain
  auto guardChain =
      MO::ChainCreateOp::create(rewriter, op->getLoc(), ValueRange{allChains});

  auto guardOp = MO::GuardOp::create(rewriter, op->getLoc(), guardChain,
                                     op->getOperands());

  rewriter.modifyOpInPlace(op, [&guardOp, &op] {
    for (auto &&[replacement, old] :
         llvm::zip(guardOp->getResults(), op->getOpOperands())) {
      old.set(replacement);
    }
  });
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

/// Checks if an op's operands contains any mutable types: opaque or buffer.
static bool hasMutableOperand(OperandRange operands) {
  return llvm::any_of(operands, [](Value val) {
    return isa<MO::BufferType, MO::OpaqueType>(val.getType());
  });
}

bool CustomOp::isInPlace() {
  // TODO(GEX-1181): design kernel API/registration mechanism to refine this.
  return hasMutableOperand(getOperands()) && maybeGetInChain().has_value() &&
         maybeGetOutChain().has_value();
}

SmallVector<ChainValue> CustomOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> CustomOp::getOutChains() {
  return getChainValues(getResults());
}

SmallVector<OpOperand *> CustomOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getOperandsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

std::optional<ChainValue> CustomOp::maybeGetInChain() {
  if (getNumOperands() == 0)
    return std::nullopt;

  auto lastOperand = getOperands().back();
  if (auto chain = dyn_cast<ChainValue>(lastOperand))
    return chain;

  return std::nullopt;
}

std::optional<ChainValue> CustomOp::maybeGetOutChain() {
  if (getNumResults() == 0)
    return std::nullopt;

  auto finalResult = getResults().back();
  if (auto chain = dyn_cast<ChainValue>(finalResult))
    return chain;

  return std::nullopt;
}

mlir::SmallVector<Value> CustomOp::getBufferOperands() {
  SmallVector<Value> result;

  for (Value val : getOperands()) {
    if (isa<MO::BufferType>(val.getType()))
      result.push_back(val);
  }

  return result;
}

void CustomOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  bool isInPlace = this->isInPlace();

  // The mutable spec isn't a required attr at the moment. If it's not set
  // conservatively assume that there's a read and write effect on all mutable
  // operands.
  if (isInPlace) {
    for (auto &&opOperand : getOperation()->getOpOperands()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    }
  }

  // Handle generic parameterization logic.
  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

//===----------------------------------------------------------------------===//
// MO Mutable tensor operations
//===----------------------------------------------------------------------===//

void BufferTransferOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable());

  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable());
}

void MutableLoadOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInBufferMutable());
}

void MutableStoreOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

void MutableStoreSliceOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

LogicalResult MutableStoreSliceOp::verify() {

  // Copied wholesale from the mo.slice verifier
  TensorType inputTy = castFromTensorLike(getInBuffer().getType());
  TensorType startTy = getStart().getType();
  TensorType stopTy = getStop().getType();
  TensorType stepTy = getStep().getType();
  int64_t inputRank = inputTy.getRank();
  if (startTy.hasStaticShape() && startTy.getIntDim(0) != inputRank)
    return emitError("start shape must be [inputRank] if it's static");

  if (stopTy.hasStaticShape() && stopTy.getIntDim(0) != inputRank)
    return emitError("stop shape must be [inputRank] if it's static");

  if (stepTy.hasStaticShape() && stepTy.getIntDim(0) != inputRank)
    return emitError("step shape must be [inputRank] if it's static");

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceSumOp
//===----------------------------------------------------------------------===//

LogicalResult AllReduceSumOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("allreduce requires at least one input");

  // Zero (the attribute default) means all devices form one group.
  int64_t ngpus = inputs.size();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (groupSize < 0)
    return emitError("group_size must be non-negative");
  if (groupSize == 0)
    groupSize = ngpus;
  if (ngpus % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        ngpus, groupSize));

  // Groups may differ in shape (e.g. per-DP-replica row counts). Inputs
  // differ in device (one per GPU), so AllTypesMatch can't be used.
  // Output-input type matching and signal buffer count are TableGen traits.
  auto firstInputType = getInputTensor(0).getType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
  }
  for (int64_t groupStart = 0; groupStart < ngpus; groupStart += groupSize) {
    auto groupInputType = getInputTensor(groupStart).getType();
    for (int64_t i = groupStart + 1, e = groupStart + groupSize; i < e; ++i) {
      auto inputType = getInputTensor(i).getType();
      if (inputType.getShape() != groupInputType.getShape())
        return emitError(
            "all inputs in each allreduce group must have the same shape");
    }
  }

  return success();
}

void AllReduceSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// BundledExpandOp
//===----------------------------------------------------------------------===//

LogicalResult BundledExpandOp::verify() {
  auto parallelOp = (*this)->getParentOfType<ParallelOp>();
  if (!parallelOp)
    return emitOpError("must be inside an mo.parallel region");

  size_t numLaunches = parallelOp.getNumLaunches();
  if (getNumResults() != numLaunches)
    return emitOpError("expected ")
           << numLaunches << " results (one per launch), got "
           << getNumResults();

  SmallVector<DeviceRefAttr> launchDevices = parallelOp.getLaunchDeviceList();
  for (auto [i, resultType] : llvm::enumerate(getResultTypes())) {
    if (!typesMatchIgnoringDeviceId(resultType, getInput().getType()))
      return emitOpError("result #")
             << i << " type " << resultType
             << " does not match input type (ignoring device) "
             << getInput().getType();

    auto expectedDevice = launchDevices[i];
    auto actualDevice = cast<TensorType>(resultType).getDeviceRef();
    if (actualDevice != expectedDevice)
      return emitOpError("result #")
             << i << " device " << actualDevice
             << " does not match launch device " << expectedDevice;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BundledAllReduceSumOp
//===----------------------------------------------------------------------===//

void BundledAllReduceSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// BundledBroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BundledBroadcastOp::verify() {
  auto signalBuffers = getSignalBuffers();
  if (signalBuffers.empty())
    return emitError("bundled broadcast requires at least one signal buffer");

  int64_t root = getRoot();
  int64_t numDevices = signalBuffers.size();
  if (root < 0 || root >= numDevices)
    return emitError(llvm::formatv(
        "root index {0} out of range for {1} devices", root, numDevices));

  auto inputType = cast<TensorType>(getInput().getType());
  // The input is the root's copy of the tensor, so it must already live on
  // the root's device. Devices can still be unassigned in graphs built in
  // memory, so only compare when both are known.
  auto rootDevice =
      cast<BufferType>(signalBuffers[root].getType()).getDeviceRef();
  auto inputDevice = inputType.getDeviceRef();
  if (inputDevice && rootDevice && inputDevice != rootDevice)
    return emitError(llvm::formatv(
        "the input must be on the root's device, {0}, but is on {1}",
        rootDevice, inputDevice));
  auto outputType = cast<TensorType>(getOutput().getType());
  if (outputType.getShape() != inputType.getShape())
    return emitError("the output must have the same shape as the input");
  if (outputType.getElementType() != inputType.getElementType())
    return emitError("the output must have the same element type as the input");
  return success();
}

void BundledBroadcastOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// BundledAllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult BundledAllGatherOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.empty())
    return emitError("bundled allgather requires at least one input");
  if (outputs.size() != inputs.size())
    return emitError(llvm::formatv(
        "number of outputs must match number of inputs (one gathered piece "
        "per peer); expected {0} outputs but got {1}",
        inputs.size(), outputs.size()));

  auto device = cast<TensorType>(outputs[0].getType()).getDeviceRef();
  for (auto [i, pair] : llvm::enumerate(llvm::zip(inputs, outputs))) {
    auto [input, output] = pair;
    auto inputType = cast<TensorType>(input.getType());
    auto outputType = cast<TensorType>(output.getType());
    if (outputType.getShape() != inputType.getShape())
      return emitError(
          llvm::formatv("output {0} must have the same shape as input {0}", i));
    if (outputType.getElementType() != inputType.getElementType())
      return emitError(llvm::formatv(
          "output {0} must have the same element type as input {0}", i));
    if (outputType.getDeviceRef() != device)
      return emitError(
          llvm::formatv("output {0} must be on this device's output device, "
                        "{1}, but is on {2}",
                        i, device, outputType.getDeviceRef()));
  }
  return success();
}

void BundledAllGatherOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// BundledAllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

void Composite::BundledAllReduceAddRMSNormQuantFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// MegaFFNNvfp4Op
//===----------------------------------------------------------------------===//

void Composite::MegaFFNNvfp4Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // The fused kernel writes the persistent `arrival_count` pool-slot buffer in
  // place (cross-CTA scheduler hand-off). Surfacing the Write lets the
  // buffer/chain machinery sequence the in-place mutation.
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getArrivalCountMutable());
}

//===----------------------------------------------------------------------===//
// DistributedScatterOp
//===----------------------------------------------------------------------===//

LogicalResult DistributedScatterOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("scatter requires at least one input");

  auto outputs = getOutputs();
  if (outputs.size() != inputs.size())
    return emitError("number of outputs must match number of inputs");

  if (getSignalBuffers().size() != inputs.size())
    return emitError("number of signal buffers must match number of inputs");

  auto firstInputType = getInputTensor(0).getType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
  }

  int64_t root = getRoot();
  if (root < 0 || root >= static_cast<int64_t>(inputs.size()))
    return emitError("root must be in range [0, ngpus)");

  return success();
}

void DistributedScatterOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// ReduceScatterSumOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterSumOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("reducescatter requires at least one input");

  // Verify input ranks and element types. Shapes are checked per
  // reduce-scatter group below.
  auto firstInputType = getInputTensor(0).getType();
  int64_t rank = firstInputType.getRank();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
    if (inputType.getRank() != firstInputType.getRank())
      return emitError("all inputs must have the same rank");
  }

  auto outputs = getOutputs();
  if (outputs.size() != inputs.size())
    return emitError("number of outputs must match number of inputs");

  if (getSignalBuffers().size() != inputs.size())
    return emitError("number of signal buffers must match number of inputs");

  int64_t ngpus = inputs.size();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (groupSize < 1)
    return emitError("group_size must be at least 1");
  if (ngpus % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        ngpus, groupSize));

  // Resolve scatter axis (negative indexing supported).
  int64_t axis = normalizeAxis(getAxis(), rank);

  if (axis < 0 || axis >= rank)
    return emitError(
        llvm::formatv("axis {0} out of bounds for rank {1}", getAxis(), rank));

  // Inputs in a group reduce together and therefore must have matching shapes.
  for (int64_t groupStart = 0; groupStart < ngpus; groupStart += groupSize) {
    auto groupInputType = getInputTensor(groupStart).getType();
    for (int64_t i = groupStart + 1, e = groupStart + groupSize; i < e; ++i) {
      auto inputType = getInputTensor(i).getType();
      if (inputType.getShape() != groupInputType.getShape())
        return emitError(
            "all inputs in each reduce-scatter group must have the same shape");
    }
  }

  for (int64_t devIdx = 0, e = outputs.size(); devIdx < e; ++devIdx) {
    auto outputType = getOutputTensor(devIdx).getType();
    auto inputType = getInputTensor(devIdx).getType();

    if (outputType.getElementType() != firstInputType.getElementType())
      return emitError("output element type must match input element type");

    if (static_cast<int64_t>(outputType.getRank()) != rank)
      return emitError("output rank must match input rank");

    for (int64_t i = 0; i < rank; ++i) {
      int64_t inputDim = inputType.getIntDim(i);
      int64_t outputDim = outputType.getIntDim(i);

      if (mlir::ShapedType::isDynamic(inputDim) ||
          mlir::ShapedType::isDynamic(outputDim))
        continue;

      if (i == axis) {
        // Scatter axis: verify partitioned size
        int64_t base = inputDim / groupSize;
        int64_t remainder = inputDim % groupSize;
        int64_t localRank = devIdx % groupSize;
        int64_t expectedSize = base + (localRank < remainder ? 1 : 0);

        if (outputDim != expectedSize)
          return emitError(llvm::formatv(
              "output dimension {0} should be {1} for device {2} "
              "(input dim {3} / group_size {4}), but got {5}",
              i, expectedSize, devIdx, inputDim, groupSize, outputDim));
      } else {
        // Non-scatter axes: must match exactly
        if (outputDim != inputDim)
          return emitError(llvm::formatv(
              "output dimension {0} must match input ({1} vs {2})", i, inputDim,
              outputDim));
      }
    }
  }

  return success();
}

void ReduceScatterSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// EPDispatchOp (BF16)
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

void EPDispatchOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();
  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

//===----------------------------------------------------------------------===//
// EPDispatchFP8Op
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchFP8Op::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch.fp8 requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

void EPDispatchFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();
  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

//===----------------------------------------------------------------------===//
// EPDispatchBlockScaledNVOp
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchBlockScaledNVOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError(
        "ep.dispatch.block.scaled.nv requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getInputScales().size() != numDevices)
    return emitError("inputScales count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getScalesOffsets().size() != numDevices)
    return emitError("scalesOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

void EPDispatchBlockScaledNVOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();

  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

//===----------------------------------------------------------------------===//
// EPDispatchBlockScaledNVDynGlobalScalesOp
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchBlockScaledNVDynGlobalScalesOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch.block.scaled.nv.dyn_global_scales requires "
                     "at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getOutputRowwiseScales().size() != numDevices)
    return emitError("outputRowwiseScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getScalesOffsets().size() != numDevices)
    return emitError("scalesOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

void EPDispatchBlockScaledNVDynGlobalScalesOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();

  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

//===----------------------------------------------------------------------===//
// EPDispatchMXFP4Op
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchMXFP4Op::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch.mxfp4 requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

void EPDispatchMXFP4Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();

  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

//===----------------------------------------------------------------------===//
// EPCombineOp
//===----------------------------------------------------------------------===//

LogicalResult EPCombineOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.combine requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getRouterWeights().size() != numDevices)
    return emitError("routerWeights count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");

  return success();
}

void EPCombineOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();
  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

// Composite::DistributedMatmulReduceScatterSumOp
//===----------------------------------------------------------------------===//

LogicalResult Composite::DistributedMatmulReduceScatterSumOp::verify() {
  auto inputsA = getInputsA();
  auto inputsB = getInputsB();
  auto outputs = getOutputs();
  auto signalBuffers = getSignalBuffers();

  if (inputsA.empty())
    return emitError("matmul_reduce_scatter requires at least one device");

  if (inputsB.size() != inputsA.size())
    return emitError(llvm::formatv(
        "number of B inputs ({0}) must match number of A inputs ({1})",
        inputsB.size(), inputsA.size()));

  if (outputs.size() != inputsA.size())
    return emitError(llvm::formatv(
        "number of outputs ({0}) must match number of inputs ({1})",
        outputs.size(), inputsA.size()));

  if (signalBuffers.size() != inputsA.size())
    return emitError(llvm::formatv(
        "number of signal buffers ({0}) must match number of inputs ({1})",
        signalBuffers.size(), inputsA.size()));

  auto firstAType = cast<TensorType>(inputsA[0].getType());
  if (!firstAType.hasRank() || firstAType.getRank() != 2)
    return emitError("A inputs must be rank-2 tensors");

  for (size_t i = 1; i < inputsA.size(); ++i) {
    auto aType = cast<TensorType>(inputsA[i].getType());
    if (aType.getShape() != firstAType.getShape())
      return emitError("all A inputs must have the same shape");
    if (aType.getElementType() != firstAType.getElementType())
      return emitError("all A inputs must have the same element type");
  }

  auto firstBType = cast<TensorType>(inputsB[0].getType());
  if (!firstBType.hasRank() || firstBType.getRank() != 2)
    return emitError("B inputs must be rank-2 tensors");

  for (size_t i = 1; i < inputsB.size(); ++i) {
    auto bType = cast<TensorType>(inputsB[i].getType());
    if (bType.getShape() != firstBType.getShape())
      return emitError("all B inputs must have the same shape");
    if (bType.getElementType() != firstBType.getElementType())
      return emitError("all B inputs must have the same element type");
  }

  // Verify matmul shape compatibility: A is [M, K], B is [N, K] (transposed).
  int64_t aK = firstAType.getIntDim(1);
  int64_t bK = firstBType.getIntDim(1);
  if (!mlir::ShapedType::isDynamic(aK) && !mlir::ShapedType::isDynamic(bK) &&
      aK != bK)
    return emitError(llvm::formatv(
        "matmul K dimension mismatch: A has K={0}, B has K={1}", aK, bK));

  if (getHasResidual()) {
    int64_t residualPeer = getResidualPeer();
    if (residualPeer < 0 ||
        residualPeer >= static_cast<int64_t>(inputsA.size()))
      return emitError(
          llvm::formatv("residual_peer ({0}) must be in range [0, {1})",
                        residualPeer, inputsA.size()));

    auto residualType = cast<TensorType>(getResidual().getType());
    if (!residualType.hasRank() || residualType.getRank() != 2)
      return emitError("residual must be a rank-2 tensor");

    // The residual must live on the same device as the peer that applies it,
    // since that peer's kernel reads the residual tensor directly.
    auto residualDevice = residualType.getDeviceRef();
    auto peerDevice =
        cast<TensorType>(inputsA[residualPeer].getType()).getDeviceRef();
    if (residualDevice && peerDevice && residualDevice != peerDevice)
      return emitError(llvm::formatv(
          "residual device must match the device of input {0}", residualPeer));

    // Residual shape must match matmul output: [M, N].
    int64_t aM = firstAType.getIntDim(0);
    int64_t bN = firstBType.getIntDim(0);
    int64_t rM = residualType.getIntDim(0);
    int64_t rN = residualType.getIntDim(1);
    if (!mlir::ShapedType::isDynamic(aM) && !mlir::ShapedType::isDynamic(rM) &&
        aM != rM)
      return emitError(llvm::formatv(
          "residual M dimension ({0}) must match A M dimension ({1})", rM, aM));
    if (!mlir::ShapedType::isDynamic(bN) && !mlir::ShapedType::isDynamic(rN) &&
        bN != rN)
      return emitError(llvm::formatv(
          "residual N dimension ({0}) must match B N dimension ({1})", rN, bN));

    if (residualType.getElementType() != firstAType.getElementType())
      return emitError("residual element type must match A element type");
  }

  return success();
}

void Composite::DistributedMatmulReduceScatterSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  auto outputs = getOutputs();
  auto inputs = getInputs();
  auto signalBuffers = getSignalBuffers();

  if (signalBuffers.size() != inputs.size())
    return emitError(
        llvm::formatv("number of signal buffers should match number of inputs. "
                      "Expected {0} signal buffers but got {1}",
                      inputs.size(), signalBuffers.size()));

  int64_t numInputs = inputs.size();
  int64_t groupSize = getGroupSize();
  if (groupSize < 1)
    return emitError("group_size must be at least 1");
  if (numInputs % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        numInputs, groupSize));

  int64_t expectedOutputs = numInputs * groupSize;
  if (static_cast<int64_t>(outputs.size()) != expectedOutputs)
    return emitError(llvm::formatv(
        "number of outputs must be num_inputs * group_size for all-gather; "
        "expected {0} outputs, found {1} (num_inputs={2}, group_size={3})",
        expectedOutputs, outputs.size(), inputs.size(), groupSize));
  return success();
}

void AllGatherOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedBroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult DistributedBroadcastOp::verify() {
  auto outputs = getOutputs();
  auto signalBuffers = getSignalBuffers();

  if (signalBuffers.empty())
    return emitError("broadcast requires at least one signal buffer");

  if (outputs.size() != signalBuffers.size())
    return emitError(
        llvm::formatv("number of outputs must match number of signal buffers. "
                      "Expected {0} outputs but got {1}",
                      signalBuffers.size(), outputs.size()));

  int64_t root = getRoot();
  int64_t numDevices = signalBuffers.size();
  if (root < 0 || root >= numDevices)
    return emitError(llvm::formatv(
        "root index {0} out of range for {1} devices", root, numDevices));

  auto inputType = cast<TensorType>(getInput().getType());
  // The input is the root's copy of the tensor, so it must already live on
  // the root's device. Devices can still be unassigned in graphs built in
  // memory, so only compare when both are known.
  auto rootDevice =
      cast<BufferType>(signalBuffers[root].getType()).getDeviceRef();
  auto inputDevice = inputType.getDeviceRef();
  if (inputDevice && rootDevice && inputDevice != rootDevice)
    return emitError(llvm::formatv(
        "the input must be on the root's device, {0}, but is on {1}",
        rootDevice, inputDevice));

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto outputType = getOutputTensor(i).getType();
    if (outputType.getShape() != inputType.getShape())
      return emitError("all outputs must have the same shape as the input");
    if (outputType.getElementType() != inputType.getElementType())
      return emitError(
          "all outputs must have the same element type as the input");
  }

  return success();
}

void DistributedBroadcastOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// AllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

void Composite::AllReduceAddRMSNormQuantFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedReduceScatterRMSNormOp
//===----------------------------------------------------------------------===//

void Composite::DistributedReduceScatterRMSNormOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormOp
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherRMSNormOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormQuantMXFP8Op
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherRMSNormQuantMXFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormQuantMXFP6Op
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherRMSNormQuantMXFP6Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormRoPEOp
//===----------------------------------------------------------------------===//

LogicalResult Composite::ReduceRMSNormRoPEOp::verify() {
  auto inputTy = cast<TensorType>(getInput().getType());
  if (!inputTy.getDeviceRef())
    return emitOpError("input must be on a device");
  if (inputTy.getDeviceRef().getLabel() != M::kGPULabel)
    return emitOpError("mo.composite.rms_norm_rope is only supported on GPU");
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Needed because the type and attribute constraints in the generated file are
// not placed in the correct namespace by tablegen.
using namespace Shape;
using M::KGEN::parseStringParam;
using M::KGEN::printStringParam;

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MODialect/MO.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GraphCompiler/MODialect/MO.cpp.inc"
