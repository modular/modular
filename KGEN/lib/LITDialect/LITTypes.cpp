//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
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

  auto *dialect = getContext()->getOrLoadDialect<KGENDialect>();
  dialect->registerMnemonicType<MetaTypeType>();
}

//===----------------------------------------------------------------------===//
// TypeSignatureType
//===----------------------------------------------------------------------===//

static ParseResult
parseTypeSignature(AsmParser &p, SmallVectorImpl<Type> &paramTypes,
                   SmallVectorImpl<StringAttr> &paramNames,
                   SmallVectorImpl<PassingKind> &paramPassingKinds,
                   SmallVectorImpl<TypedAttr> &defaultParameters,
                   bool &paramVarArg) {
  SmallVector<Type> resultParamTypes;
  if (parseOptionalParamSignature(p, paramTypes, resultParamTypes, paramNames,
                                  paramPassingKinds, defaultParameters))
    return failure();
  if (!resultParamTypes.empty())
    return p.emitError(p.getCurrentLocation(),
                       "unexpected result parameters for type signature");
  paramVarArg = succeeded(p.parseOptionalKeyword("param_vararg"));
  return success();
}

static void printTypeSignature(AsmPrinter &p, ArrayRef<Type> paramTypes,
                               ArrayRef<StringAttr> paramNames,
                               ArrayRef<PassingKind> paramPassingKinds,
                               ArrayRef<TypedAttr> defaultParameters,
                               bool paramVarArg) {
  printOptionalParamSignature(p, paramTypes, /*resultParamTypes=*/{},
                              paramNames, paramPassingKinds, defaultParameters);
  if (paramVarArg)
    p << " param_vararg";
}

LogicalResult TypeSignatureType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<Type> paramTypes,
    ArrayRef<StringAttr> paramNames, ArrayRef<PassingKind> paramPassingKinds,
    ArrayRef<TypedAttr> defaultParameters, bool paramVarArg) {
  if (paramNames.size() != paramPassingKinds.size()) {
    return emitError()
           << "number of parameter names and passing kinds must match";
  }
  for (StringAttr name : paramNames)
    if (!name)
      return emitError() << "parameter name cannot be null";
  if (paramNames.size() != paramTypes.size()) {
    return emitError() << "number of parameter names doesn't match number of "
                          "input parameter types";
  }
  if (defaultParameters.size() > paramTypes.size()) {
    return emitError() << "there are more default parameters than parameters: "
                       << defaultParameters.size() << " vs. "
                       << paramTypes.size();
  }
  for (auto [defaultsIndex, value] : llvm::enumerate(defaultParameters)) {
    size_t index = paramTypes.size() - defaultParameters.size() + defaultsIndex;
    Type expected = paramTypes[index];
    if (value.getType() != expected &&
        !llvm::isa<TypeCheckErrorType>(expected)) {
      return emitError() << "parameter #" << index << " has type " << expected
                         << " but default parameter has type "
                         << value.getType();
    }
  }
  if (paramVarArg) {
    if (paramTypes.empty()) {
      return emitError() << "type signature with 'param_vararg' must have at "
                            "least one parameter";
    }
    if (!::isa<VariadicType>(paramTypes.back()))
      return emitError() << "expected last parameter type to be a variadic "
                            "type for 'param_vararg'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MetaTypeType
//===----------------------------------------------------------------------===//

OptionalParseResult MetaTypeType::parseValue(AsmParser &p,
                                             TypedAttr &value) const {
  Type type;
  OptionalParseResult result = parseOptionalKGENType(p, type);
  if (!result.has_value())
    return {};
  if (failed(*result))
    return failure();
  value = TypeConstantAttr::get(type, *this);
  return mlir::success();
}

LogicalResult MetaTypeType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto type = ::dyn_cast<TypeConstantAttr>(value);
  if (!type)
    return failure();
  printKGENType(p, type.getValue());
  return success();
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
  llvm::SMLoc startLoc = p.getCurrentLocation();
  SmallVector<Type> inputParamTypes, resultParamTypes;
  SmallVector<TypedAttr> defaultParamValues;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  if (failed(parseOptionalParamSignature(p, inputParamTypes, resultParamTypes,
                                         paramNames, paramPassingKinds,
                                         defaultParamValues)))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> argDefaults;
  SmallVector<ValueInputConvention> inputConventions;

  StarSlashParser ssParser(p);
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res =
            ssParser.parseOptionalStarSlash(p.getCurrentLocation());
        res.has_value())
      return res.value();

    // Parse an optional argument name.
    if (parseOptionalName(p, argNames.emplace_back()))
      return failure();

    // Parse the argument type and its input convention.
    Type &type = argTypes.emplace_back();
    if (p.parseType(type) ||
        parseInputConvention(p, inputConventions.emplace_back(),
                             ValueInputConvention::OwnedInReg))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(
            p, defaultVal, type,
            SignatureType::hasAddress(inputConventions.back()))))
      return failure();
    if (defaultVal)
      argDefaults.emplace_back(defaultVal);

    return success();
  };

  FunctionType functionType;
  FnEffects effects;
  if (parseSignatureValues(p, parseArg, functionType, effects,
                           /*optionalResultList=*/false))
    return failure();

  auto [numPosOnly, numPosOrKw, numKwOnly] = ssParser.getNumPassingKinds();
  SmallVector<PassingKind> argPassingKinds(numPosOnly, PassingKind::PosOnly);
  argPassingKinds.append(numPosOrKw, PassingKind::PosOrKw);
  argPassingKinds.append(numKwOnly, PassingKind::KwOnly);

  MLIRContext *ctx = p.getContext();
  signature = SignatureType::getChecked(
      [&] { return p.emitError(startLoc); }, functionType, inputParamTypes,
      resultParamTypes, inputConventions, effects,
      FnMetadataAttr::get(ctx, argNames, argPassingKinds, paramNames,
                          paramPassingKinds, argDefaults, defaultParamValues));
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

  p.emitError(typeLoc) << "unknown type `" << mnemonic << "` in dialect `"
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
  printOptionalParamSignature(
      p, signature.getInputParamTypes(), signature.getResultParamTypes(),
      signature.getParamNames(), signature.getParamPassingKinds(),
      signature.getMetadata().getDefaultParameters());

  ArrayRef<TypedAttr> defaultArgs = signature.getDefaultArguments();
  size_t numInputs = signature.getNumInputs();
  size_t defaultIndex = numInputs - defaultArgs.size();

  StarSlashPrinter ssPrinter(p, numInputs, '|');
  auto printElt = [&](unsigned i) {
    ssPrinter.printOptionalStarSlash(signature.getArgPassingKinds()[i], i);

    StringAttr argName = signature.getArgName(i);
    if (!argName.empty()) {
      p.printString(argName);
      p << ": ";
    }

    p << signature.getValueInputs()[i];

    printInputConvention(p, signature.getInputConvention(i),
                         ValueInputConvention::OwnedInReg);
    if (i >= defaultIndex) {
      p << " = ";
      printParamValue(p, defaultArgs[i - defaultIndex]);
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    ssPrinter.printOptionalTrailingSlash(i);
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

ArrayRef<PassingKind> LITSignatureType::getArgPassingKinds() {
  return getMetadata().getArgPassingKinds();
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

ArrayRef<PassingKind> LITSignatureType::getParamPassingKinds() {
  return getMetadata().getParamPassingKinds();
}

LITSignatureType LITSignatureType::dropParamValues() {
  auto metadata =
      FnMetadataAttr::get(getContext(), getArgNames(), getArgPassingKinds(),
                          /*paramNames=*/{}, /*paramPassingKinds=*/{},
                          getDefaultArguments(), /*defaultParameters=*/{});
  return get(getValues(), /*inputParamTypes=*/{}, getResultParamTypes(),
             getInputConventions(), getFnEffects(), metadata);
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
  auto funcType = FunctionType::get(ctx, inputs, results);

  size_t numInputs = funcType.getNumInputs();
  SmallVector<StringAttr> argNames(numInputs, StringAttr::get(ctx));
  SmallVector<PassingKind> argPassingKinds(numInputs, PassingKind::PosOnly);
  auto metadata = FnMetadataAttr::get(ctx, argNames, argPassingKinds);
  return LITSignatureType::get(funcType, /*inputParamTypes=*/{},
                               /*resultParamTypes=*/{},
                               /*convs=*/{}, /*effects=*/{}, metadata);
}

LITSignatureType LITSignatureType::get(FunctionType values,
                                       ArrayRef<Type> inputParamTypes,
                                       ArrayRef<Type> resultParamTypes,
                                       ArrayRef<ValueInputConvention> convs,
                                       FnEffects effects, Attribute metadata) {
  assert(metadata && "LITSignatureType must have non-null metadata");
  return SignatureType::get(values, inputParamTypes, resultParamTypes, convs,
                            effects, metadata);
}
