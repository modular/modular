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
// This file implements the assembly format of the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
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

using namespace M;
using namespace MO;

// Needed because the type and attribute constraints in the generated file
// are not placed in the correct namespace by tablegen. They feed the
// GET_OP_CLASSES include at the end of this file.
using namespace Shape;
using M::KGEN::parseStringParam;
using M::KGEN::printStringParam;

namespace M::MO {

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

ParseResult parseAllGatherAttrs(OpAsmParser &p, IntegerAttr &groupSize,
                                BoolAttr &sliceRows) {
  Builder &b = p.getBuilder();
  return parseAttrGroup(
      p, {{"group_size",
           [&] { return p.parseAttribute(groupSize, b.getIntegerType(64)); }},
          {"slice_rows",
           [&] { return p.parseAttribute(sliceRows, b.getIntegerType(1)); }}});
}

void printAllGatherAttrs(OpAsmPrinter &p, Operation *, IntegerAttr groupSize,
                         BoolAttr sliceRows) {
  printAttrGroup(p, {nonDefault("group_size", groupSize, 0),
                     nonDefault("slice_rows", sliceRows, false)});
}

ParseResult parseAllGatherConcatAttrs(OpAsmParser &p, IntegerAttr &axis,
                                      IntegerAttr &groupSize) {
  Type i64 = p.getBuilder().getIntegerType(64);
  return parseAttrGroup(
      p, {{"axis", [&] { return p.parseAttribute(axis, i64); }},
          {"group_size", [&] { return p.parseAttribute(groupSize, i64); }}});
}

void printAllGatherConcatAttrs(OpAsmPrinter &p, Operation *, IntegerAttr axis,
                               IntegerAttr groupSize) {
  printAttrGroup(p, {nonDefault("axis", axis, -1),
                     nonDefault("group_size", groupSize, 0)});
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
                                           BoolAttr &hasResidual,
                                           IntegerAttr &quantBits) {
  Builder &b = p.getBuilder();
  return parseAttrGroup(
      p, {{"group_size",
           [&] { return p.parseAttribute(groupSize, b.getIntegerType(64)); }},
          {"has_residual",
           [&] { return p.parseAttribute(hasResidual, b.getIntegerType(1)); }},
          {"quant_bits",
           [&] { return p.parseAttribute(quantBits, b.getIntegerType(64)); }}});
}

void printReduceScatterRMSNormAttrs(OpAsmPrinter &p, Operation *,
                                    IntegerAttr groupSize, BoolAttr hasResidual,
                                    IntegerAttr quantBits) {
  printAttrGroup(p, {nonDefault("group_size", groupSize, 0),
                     nonDefault("has_residual", hasResidual, false),
                     nonDefault("quant_bits", quantBits, 0)});
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

ParseResult parseGroupedMatmulSwigluNvfp4Attrs(
    OpAsmParser &p, StringAttr &activation, ElementsAttr &swigluAlpha,
    ElementsAttr &swigluLimit, BoolAttr &hasARowScales, IntegerAttr &maxTotalM,
    BoolAttr &prefillDynamicClaim) {
  Type i1 = p.getBuilder().getIntegerType(1);
  Type i64 = p.getBuilder().getIntegerType(64);
  return parseAttrGroup(
      p, {{"activation", [&] { return p.parseAttribute(activation); }},
          {"swiglu_alpha", [&] { return p.parseAttribute(swigluAlpha); }},
          {"swiglu_limit", [&] { return p.parseAttribute(swigluLimit); }},
          {"has_a_row_scales",
           [&] { return p.parseAttribute(hasARowScales, i1); }},
          {"max_total_m", [&] { return p.parseAttribute(maxTotalM, i64); }},
          {"prefill_dynamic_claim",
           [&] { return p.parseAttribute(prefillDynamicClaim, i1); }}});
}

void printGroupedMatmulSwigluNvfp4Attrs(
    OpAsmPrinter &p, Operation *, StringAttr activation,
    ElementsAttr swigluAlpha, ElementsAttr swigluLimit, BoolAttr hasARowScales,
    IntegerAttr maxTotalM, BoolAttr prefillDynamicClaim) {
  printAttrGroup(
      p, {nonDefault("activation", activation, "silu"),
          {"swiglu_alpha", swigluAlpha},
          {"swiglu_limit", swigluLimit},
          nonDefault("has_a_row_scales", hasARowScales, false),
          nonDefault("max_total_m", maxTotalM, 0),
          nonDefault("prefill_dynamic_claim", prefillDynamicClaim, false)});
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

#define GET_OP_CLASSES
#include "GraphCompiler/MODialect/MO.cpp.inc"
