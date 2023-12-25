//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/KGENDialect/KGENParameters.h"
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
  dialect->registerMnemonicType<TraitType>();
  dialect->registerMnemonicType<LifetimeType>();
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

TypeSignatureType TypeSignatureType::remapToSignature(
    function_ref<InFlightDiagnostic()> emitError, ParamDeclArrayAttr paramDecls,
    ArrayRef<StringAttr> paramNames, ArrayRef<PassingKind> passingKinds,
    ArrayRef<TypedAttr> defaults, bool paramVarArg) {
  IndexRefRemapper remapper(paramDecls, {});
  SmallVector<Type> inputParamTypes =
      llvm::map_to_vector(paramDecls, [&](ParamDeclAttr decl) {
        return remapper.replace(decl.getType());
      });
  return TypeSignatureType::getChecked(
      emitError, paramDecls.getContext(), inputParamTypes, paramNames,
      passingKinds, remapper.replace(ArrayRef(defaults)), paramVarArg);
}

//===----------------------------------------------------------------------===//
// MetaTypeType
//===----------------------------------------------------------------------===//

static OptionalParseResult parseTypeValue(AsmParser &p, TypedAttr &value,
                                          Type metatype) {
  Type typeValue;
  bool parsingVTable = succeeded(p.parseOptionalLSquare());
  auto vtable = VTableAttr::get(p.getContext(), {});

  SymbolRefAttr symbol;
  OptionalParseResult result = p.parseOptionalAttribute(symbol);
  if (result.has_value()) {
    if (failed(*result))
      return failure();
    SmallVector<TypedAttr> values;
    if (parseParameterValues(p, values))
      return failure();
    Type declRefMetaType = metatype;
    if (succeeded(p.parseOptionalColon()))
      if (parseKGENType(p, declRefMetaType))
        return failure();
    typeValue = DeclRefType::get(symbol, values, declRefMetaType);
  } else {
    result = parseOptionalKGENType(p, typeValue);
    if (!result.has_value()) {
      // If a '[' was seen, require a type to be present.
      if (parsingVTable)
        return p.emitError(p.getCurrentLocation(), "expected a type");
      return {};
    }
    if (failed(*result))
      return failure();
  }

  // Parse the vtable if a '[' was seen.
  if (parsingVTable) {
    if (p.parseComma() || p.parseLBrace() ||
        (p.parseOptionalRBrace() &&
         (!(vtable = cast_or_null<VTableAttr>(VTableAttr::parse(p, {}))) ||
          p.parseRBrace())) ||
        p.parseRSquare())
      return failure();
  }

  value = TypeConstantAttr::get(typeValue, metatype, vtable);
  return mlir::success();
}

static LogicalResult printTypeValue(AsmPrinter &p, TypedAttr value,
                                    Type metatype) {
  auto type = dyn_cast<TypeConstantAttr>(value);
  if (!type)
    return failure();

  VTableAttr vtable = type.getVTable();
  if (!vtable.getEntries().empty())
    p << '[';

  if (auto ref = ::dyn_cast<DeclRefType>(type.getValue())) {
    // Use the alias printer if suitable.
    if (ref.getAliasName()) {
      p.printType(ref);
    } else {
      p << ref.getSymbol();
      printParameterValues(p, ref.getParamValues());
      if (ref.getMetaType() != metatype) {
        p << " : ";
        printKGENType(p, ref.getMetaType());
      }
    }
  } else {
    printKGENType(p, type.getValue());
  }

  if (!vtable.getEntries().empty()) {
    p << ", {";
    vtable.print(p);
    p << "}]";
  }
  return success();
}

OptionalParseResult MetaTypeType::parseValue(AsmParser &p,
                                             TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult MetaTypeType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value, *this);
}

MetaTypeType MetaTypeType::bind(ArrayRef<TypedAttr> values) const {
  assert(getParamValues().size() == values.size() && "expected full value set");

  TypeSignatureType sig = getSignature();
  size_t defaultIdx =
      sig.getNumInputParams() - sig.getDefaultParameters().size();

  auto sigRange = llvm::enumerate(sig.getInputParamTypes(), sig.getParamNames(),
                                  sig.getParamPassingKinds());
  auto sigIt = sigRange.begin();

  SmallVector<Type> newParamTypes;
  SmallVector<StringAttr> newParamNames;
  SmallVector<PassingKind> newPassingKinds;
  SmallVector<TypedAttr> newDefaults;
  bool paramVarArg = false;

  for (auto [cur, val] : llvm::zip(getParamValues(), values)) {
    // Current value is unbound. This corresponds to a parameter in the
    // signature.
    if (::isa<UnboundAttr>(cur)) {
      if (::isa<UnboundAttr>(val)) {
        auto [i, type, name, kind] = *sigIt;
        newParamTypes.push_back(type);
        newParamNames.push_back(name);
        newPassingKinds.push_back(kind);
        if (i >= defaultIdx)
          newDefaults.push_back(sig.getDefaultParameters()[i - defaultIdx]);
        if (sig.isVarArg(i))
          paramVarArg = true;
      }
      ++sigIt;
      continue;
    }
    assert(cur == val && "cannot change bound parameter value");
  }
  assert(sigIt == sigRange.end() && "expected signature to get processed");

  auto newSig =
      TypeSignatureType::get(getContext(), newParamTypes, newParamNames,
                             newPassingKinds, newDefaults, paramVarArg);
  return MetaTypeType::get(getSymbol(), values, newSig);
}

//===----------------------------------------------------------------------===//
// TraitType
//===----------------------------------------------------------------------===//

OptionalParseResult TraitType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult TraitType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value, *this);
}

//===----------------------------------------------------------------------===//
// LifetimeType
//===----------------------------------------------------------------------===//

OptionalParseResult LifetimeType::parseValue(AsmParser &p,
                                             TypedAttr &result) const {
  if (succeeded(p.parseOptionalStar())) {
    std::string str;
    // Resolve ambiguity with *"...".
    if (succeeded(p.parseOptionalString(&str))) {
      result = ParamDeclRefAttr::get(str, *this);
      return mlir::success();
    }

    size_t depth, index;
    if (p.parseLSquare() || p.parseInteger(depth) || p.parseComma() ||
        p.parseInteger(index) || p.parseRSquare())
      return failure();
    result = ImplicitLifetimeRefAttr::get(p.getContext(), depth, index);
    return mlir::success();
  }
  return std::nullopt;
}

LogicalResult LifetimeType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto ref = ::dyn_cast<ImplicitLifetimeRefAttr>(value)) {
    p << "*[" << ref.getDepth() << ',' << ref.getIndex() << ']';
    return success();
  }
  return failure();
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
  auto typeExpr = TypeConstantAttr::get(
      elementType, AnyRegTypeType::get(elementType.getContext()));
  return get(isMutable, typeExpr, lifetime);
}

Type RefType::getElementAsType() {
  TypedAttr elemType = getElementType();
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(elemType))
    return typeCst.getValue();
  assert(LIT::isTypeExpr(elemType) &&
         "parameter expr must be a type expression");
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

  size_t numLifetimeDecls = 0;
  if (succeeded(p.parseOptionalLSquare()))
    if (p.parseInteger(numLifetimeDecls) || p.parseRSquare())
      return failure();

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

  PassingKindParser passingKindParser(p);
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res =
            passingKindParser.parseOptionalStarSlash(p.getCurrentLocation());
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

  SmallVector<PassingKind> argPassingKinds;
  passingKindParser.populatePassingKinds(argPassingKinds);

  MLIRContext *ctx = p.getContext();
  signature = SignatureType::getChecked(
      [&] { return p.emitError(startLoc); }, functionType, inputParamTypes,
      resultParamTypes, inputConventions, effects,
      FnMetadataAttr::get(ctx, argNames, argPassingKinds, paramNames,
                          paramPassingKinds, argDefaults, defaultParamValues,
                          numLifetimeDecls));
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

  if (unsigned numLifetimeDecls = getNumImplicitLifetimeDecls())
    p << '[' << numLifetimeDecls << ']';

  printOptionalParamSignature(
      p, signature.getInputParamTypes(), signature.getResultParamTypes(),
      signature.getParamNames(), signature.getParamPassingKinds(),
      signature.getMetadata().getDefaultParameters());

  ArrayRef<TypedAttr> defaultArgs = signature.getDefaultArguments();
  size_t numInputs = signature.getNumInputs();
  size_t defaultIndex = numInputs - defaultArgs.size();

  PassingKindPrinter passingKindPrinter(p, numInputs, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(signature.getArgPassingKinds()[i],
                                              i);

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
    passingKindPrinter.printOptionalTrailingSlash(i);
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

/// Get the number of implicit lifetime decls this function type carries.
size_t LITSignatureType::getNumImplicitLifetimeDecls() {
  return getMetadata().getNumImplicitLifetimeDecls();
}

LITSignatureType LITSignatureType::dropParamValues() {
  auto metadata =
      FnMetadataAttr::get(getContext(), getArgNames(), getArgPassingKinds(),
                          /*paramNames=*/{}, /*paramPassingKinds=*/{},
                          getDefaultArguments(), /*defaultParameters=*/{},
                          /*numImplicitLifetimeDecls=*/0);
  return get(getValues(), /*inputParamTypes=*/{}, getResultParamTypes(),
             getInputConventions(), getFnEffects(), metadata);
}

FunctionType LITSignatureType::substituteImplicitLifetimesIntoValues(
    ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError) {
  return cast_or_null<FunctionType>(
      substituteImplicitLifetimes(getValues(), values, emitError));
}

/// Substitute implicit lifetime references in an attribute or type.
Type LITSignatureType::substituteImplicitLifetimes(
    Type value, ArrayRef<TypedAttr> values,
    function_ref<InFlightDiagnostic()> emitError) {
  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // If we are substituting the signature directly, subtract 1.
      if (auto ref = ::dyn_cast<ImplicitLifetimeRefAttr>(attr);
          ref && ref.getDepth() == depth) {
        // Verify if requested.
        if (ref.getIndex() >= values.size()) {
          emitError() << "implicit lifetime reference at depth " << depth
                      << " has an out-of-range index: " << ref.getIndex()
                      << " >= " << values.size();
          hadError = true;
          return ref;
        }
        return values[ref.getIndex()];
      }
      return nullptr;
    }

    ArrayRef<TypedAttr> values;
    function_ref<InFlightDiagnostic()> emitError;
    bool hadError = false;
  } substitutor;
  substitutor.values = values;
  substitutor.emitError = emitError;
  Type result = substitutor.replace(value);
  return substitutor.hadError ? Type() : result;
}

/// This method replaces direct uses of NAMED implicit lifetime declarations
/// with index-based references corresponding to the signature. `lifetimeDecls`
/// specifies the names of the implicit lifetime decls.
SignatureType LITSignatureType::replaceImplicitLifetimesWithIndexes(
    ArrayRef<ParamDeclAttr> lifetimeDecls) {
  assert(lifetimeDecls.size() == getNumImplicitLifetimeDecls() &&
         "Incorrect number of lifetime decls");

  // If there are no implicit lifetimes, then this is a noop.
  if (lifetimeDecls.empty())
    return *this;

  // Replace named implicit lifetime parameter references with index-based
  // references in the signature.
  struct LifetimeDeclRemapper : IndexParameterReplacer<LifetimeDeclRemapper> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = ::dyn_cast<ParamDeclRefAttr>(attr)) {
        if (auto it = mapping.find(ref.getName()); it != mapping.end()) {
          // Subtract 1 because we're replacing the signature directly.
          size_t index = it->second;
          return ImplicitLifetimeRefAttr::get(attr.getContext(), depth - 1,
                                              index);
        }
      }
      return nullptr;
    }

    DenseMap<StringAttr, size_t> mapping;
  } remapper;
  for (auto [i, decl] : llvm::enumerate(lifetimeDecls))
    remapper.mapping.try_emplace(decl.getName(), i);
  return remapper.replace(*this);
}

bool LITSignatureType::classof(SignatureType type) {
  return ::isa_and_nonnull<FnMetadataAttr>(type.getMetadata());
}

bool LITSignatureType::classof(Type type) {
  if (auto sig = ::dyn_cast<SignatureType>(type))
    return classof(sig);
  return false;
}

// Determine how many implicit lifetimes a signature with the specified input
// values should have.
size_t
LITSignatureType::countImplicitLifetimes(ArrayRef<ValueInputConvention> convs) {
  size_t result = 0;
  for (auto conv : convs)
#if 0 // TODO(clattner / references)
    if (conv == ValueInputConvention::ByRefResult)
      ++result;
#else
    (void)conv;
#endif
    return result;
}

LITSignatureType LITSignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                       TypeRange results,
                                       size_t numImplicitLifetimeDecls) {
  auto funcType = FunctionType::get(ctx, inputs, results);

  size_t numInputs = funcType.getNumInputs();
  SmallVector<StringAttr> argNames(numInputs, StringAttr::get(ctx));
  SmallVector<PassingKind> argPassingKinds(numInputs, PassingKind::PosOnly);
  auto metadata = FnMetadataAttr::get(ctx, argNames, argPassingKinds,
                                      numImplicitLifetimeDecls);
  return LITSignatureType::get(funcType, /*inputParamTypes=*/{},
                               /*resultParamTypes=*/{},
                               /*convs=*/{}, /*effects=*/{}, metadata);
}

LITSignatureType LITSignatureType::get(FunctionType values,
                                       ArrayRef<Type> inputParamTypes,
                                       ArrayRef<Type> resultParamTypes,
                                       ArrayRef<ValueInputConvention> convs,
                                       FnEffects effects,
                                       FnMetadataAttr metadata) {
  assert(metadata && "LITSignatureType must have non-null metadata");
  assert(countImplicitLifetimes(convs) ==
         metadata.getNumImplicitLifetimeDecls());
  return SignatureType::get(values, inputParamTypes, resultParamTypes, convs,
                            effects, metadata);
}
