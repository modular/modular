//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LITDialect
//===----------------------------------------------------------------------===//

void LITDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/LITDialect/LITTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

RefType RefType::get(bool isMutable, TypedAttr elementType,
                     TypedAttr lifetime) {
  auto *ctx = elementType.getContext();
  return get(ctx, isMutable, elementType, lifetime);
}

RefType RefType::get(bool isMutable, Type elementType, TypedAttr lifetime) {
  return get(isMutable, TypeConstantAttr::get(elementType), lifetime);
}

Type RefType::getElementAsType() {
  TypedAttr elemType = getElementType();
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(elemType))
    return typeCst.getValue();
  assert(::isa<MLIRTypeType>(elemType.getType()) &&
         "parameter expr must have metatype type");
  return ParamRefType::get(elemType);
}

/// Return the pointer type that corresponds to this reference type, ignoring
/// the lifetime and the mutability.
PointerType RefType::getAsPointerType() {
  return PointerType::get(getElementType());
}

/// Given the specified pointer type, return a reference type of the same
/// element but with a hacked lifetime.
/// TODO(references): Remove. This is just for migration.
RefType RefType::getRefForPointerHACK(PointerType type, bool isMut) {
  return RefType::get(isMut, type.getElementType(),
                      LifetimeAttr::get(type.getContext()));
}

/// Print/Parse a parameter value that is known to have `lifetime` type.
static void printLifetimeParamValue(AsmPrinter &p, TypedAttr value) {
  printParamValue(p, value);
}
static ParseResult parseLifetimeParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value,
                         LifetimeType::get(p.getBuilder().getContext()));
}

/// Print/Parse the 'mut' keyword as 1, and its absence as 0.
static void printMutFlag(AsmPrinter &p, bool value) {
  if (value)
    p << "mut ";
}
static ParseResult parseMutFlag(AsmParser &p, bool &value) {
  value = succeeded(p.parseOptionalKeyword("mut"));
  return success();
}

//===----------------------------------------------------------------------===//
// REPLResultRefType
//===----------------------------------------------------------------------===//

REPLResultRefType REPLResultRefType::get(Type elementType) {
  auto *ctx = elementType.getContext();
  return get(ctx, elementType);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// SignatureType Parsing
//===----------------------------------------------------------------------===//

static ParseResult parseLITSignature(AsmParser &p, Type &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<Type> inputParamTypes, resultParamTypes;
  SmallVector<TypedAttr> defaultParamValues;
  SmallVector<StringAttr> paramNames;
  if (failed(parseOptionalParamSignature(p, inputParamTypes, resultParamTypes,
                                         paramNames, defaultParamValues)))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> argDefaults;
  SmallVector<ValueInputConvention> inputConventions;
  auto parseElt = [&]() -> Type {
    // Parse an optional argument name.
    if (parseOptionalName(p, argNames.emplace_back()))
      return {};

    // Parse the argument type and its input convention.
    Type type;
    if (p.parseType(type) ||
        parseInputConvention(p, inputConventions.emplace_back()))
      return {};

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, type)))
      return {};
    if (defaultVal)
      argDefaults.emplace_back(defaultVal);
    return type;
  };

  FunctionType functionType;
  FnEffects effects;
  if (parseSignatureValues(p, parseElt, functionType, effects,
                           /*optionalResultList=*/false))
    return failure();
  signature = SignatureType::getChecked(
      [&] { return p.emitError(loc); }, functionType,
      TypeArrayAttr::get(p.getContext(), inputParamTypes),
      TypeArrayAttr::get(p.getContext(), resultParamTypes), inputConventions,
      effects,
      FnMetadataAttr::get(p.getContext(), argNames, paramNames, argDefaults,
                          defaultParamValues));
  return success(!!signature);
}

Type LITDialect::parseType(DialectAsmParser &p) const {
  llvm::SMLoc typeLoc = p.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  OptionalParseResult parseResult = generatedTypeParser(p, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Special alias for `!lit.signature` type.
  if (mnemonic == "signature") {
    if (p.parseLess() || parseLITSignature(p, genType) || p.parseGreater())
      return {};
    return genType;
  }

  p.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `"
                       << getNamespace() << "`";
  return {};
}

void LITDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (succeeded(generatedTypePrinter(type, p)))
    return;
}

void FnMetadataAttr::printSignature(AsmPrinter &p, SignatureType sig) const {
  p << "!lit.signature<";
  auto signature = ::cast<LITSignatureType>(sig);
  printOptionalParamSignature(p, signature.getInputParamTypes(),
                              signature.getResultParamTypes(),
                              signature.getParamNames(),
                              signature.getMetadata().getDefaultParameters());
  auto printElt = [&](unsigned i) {
    StringAttr argName = signature.getArgName(i);
    if (!argName.empty()) {
      p.printString(argName);
      p << ": ";
    }

    p << signature.getValueInputs()[i];

    printInputConvention(p, signature.getInputConvention(i));
    size_t defaultIndex =
        signature.getNumInputs() - signature.getDefaultArguments().size();
    if (i >= defaultIndex) {
      p << " = ";
      printParamValue(p, signature.getDefaultArguments()[i - defaultIndex]);
    }
  };

  printSignatureValues(p, printElt, signature.getValues(), signature,
                       /*optionalResultList=*/false);
  p << '>';
}

//===----------------------------------------------------------------------===//
// LITSignatureType
//===----------------------------------------------------------------------===//

LITSignatureType::LITSignatureType(SignatureType sig) : SignatureType(sig) {
  assert((!sig || ::isa_and_nonnull<FnMetadataAttr>(sig.getMetadata())) &&
         "expected LIT function metadata");
}

FnMetadataAttr LITSignatureType::getMetadata() {
  return ::cast<FnMetadataAttr>(SignatureType::getMetadata());
}

ArrayRef<StringAttr> LITSignatureType::getArgNames() {
  return getMetadata().getArgNames();
}

StringAttr LITSignatureType::getArgName(size_t inputNo) {
  return getArgNames()[inputNo];
}

ArrayRef<TypedAttr> LITSignatureType::getDefaultArguments() {
  return getMetadata().getDefaultArguments();
}

ArrayRef<TypedAttr> LITSignatureType::getDefaultParameters() {
  return getMetadata().getDefaultParameters();
}

ArrayRef<StringAttr> LITSignatureType::getParamNames() {
  return getMetadata().getParamNames();
}

LITSignatureType LITSignatureType::dropParamValues() {
  auto metadata = FnMetadataAttr::get(getContext(), getArgNames(), {},
                                      getDefaultArguments(), {});
  return get(getValues(), TypeArrayAttr::get(getContext(), {}),
             getResultParamTypes(), getInputConventions(), getFnEffects(),
             metadata);
}

bool LITSignatureType::classof(SignatureType type) {
  return ::isa_and_nonnull<FnMetadataAttr>(type.getMetadata());
}

bool LITSignatureType::classof(Type type) {
  if (auto sig = ::dyn_cast<SignatureType>(type))
    return classof(sig);
  return false;
}

LITSignatureType LITSignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                       TypeRange results) {
  return get(FunctionType::get(ctx, inputs, results));
}

LITSignatureType LITSignatureType::get(FunctionType values,
                                       TypeArrayAttr inputParams,
                                       TypeArrayAttr resultParams,
                                       ArrayRef<ValueInputConvention> convs,
                                       FnEffects effects, Attribute metadata) {
  if (!metadata) {
    metadata = FnMetadataAttr::get(values.getContext(), values.getNumInputs(),
                                   inputParams ? inputParams.size() : 0);
  }
  return SignatureType::get(values, inputParams, resultParams, convs, effects,
                            metadata);
}
