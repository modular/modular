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
                   SmallVectorImpl<TypedAttr> &defaultPosParams,
                   SmallVectorImpl<TypedAttr> &defaultKwOnlyParams,
                   bool &paramVarArg) {
  SmallVector<Type> resultParamTypes;
  if (parseOptionalParamSignature(p, paramTypes, resultParamTypes, paramNames,
                                  paramPassingKinds, defaultPosParams,
                                  defaultKwOnlyParams))
    return failure();
  if (!resultParamTypes.empty()) {
    return p.emitError(p.getCurrentLocation(),
                       "unexpected result parameters for type signature");
  }
  paramVarArg = succeeded(p.parseOptionalKeyword("param_vararg"));
  return success();
}

static void printTypeSignature(AsmPrinter &p, ArrayRef<Type> paramTypes,
                               ArrayRef<StringAttr> paramNames,
                               ArrayRef<PassingKind> paramPassingKinds,
                               ArrayRef<TypedAttr> defaultPosParams,
                               ArrayRef<TypedAttr> defaultKwOnlyParams,
                               bool paramVarArg) {
  printOptionalParamSignature(p, paramTypes, /*resultParamTypes=*/{},
                              paramNames, paramPassingKinds, defaultPosParams,
                              defaultKwOnlyParams);
  if (paramVarArg)
    p << " param_vararg";
}

LogicalResult TypeSignatureType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<Type> paramTypes,
    ArrayRef<StringAttr> paramNames, ArrayRef<PassingKind> paramPassingKinds,
    ArrayRef<TypedAttr> defaultPosParams,
    ArrayRef<TypedAttr> defaultKwOnlyParams, bool paramVarArg) {
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

  if (paramVarArg) {
    if (paramTypes.empty()) {
      return emitError() << "type signature with 'param_vararg' must have at "
                            "least one parameter";
    }
    if (!::isa<VariadicType>(
            paramTypes[countNumPositional(paramPassingKinds) - 1])) {
      return emitError() << "expected last positional parameter type to be a "
                            "variadic type for 'param_vararg'";
    }
  }

  return verifyDefaults(emitError, defaultPosParams, defaultKwOnlyParams,
                        paramPassingKinds, paramTypes, "parameter");
}

TypeSignatureType TypeSignatureType::remapToSignature(
    function_ref<InFlightDiagnostic()> emitError, ParamDeclArrayAttr paramDecls,
    ArrayRef<StringAttr> paramNames, ArrayRef<PassingKind> passingKinds,
    ArrayRef<TypedAttr> defaultPosParams,
    ArrayRef<TypedAttr> defaultKwOnlyParams, bool paramVarArg) {
  IndexRefRemapper remapper(paramDecls, {});
  SmallVector<Type> inputParamTypes =
      llvm::map_to_vector(paramDecls, [&](ParamDeclAttr decl) {
        return remapper.replace(decl.getType());
      });
  return TypeSignatureType::getChecked(
      emitError, paramDecls.getContext(), inputParamTypes, paramNames,
      passingKinds, remapper.replace(ArrayRef(defaultPosParams)),
      remapper.replace(ArrayRef(defaultKwOnlyParams)), paramVarArg);
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
      sig.getNumInputParams() - sig.getDefaultPosParams().size();

  auto sigRange = llvm::enumerate(sig.getInputParamTypes(), sig.getParamNames(),
                                  sig.getParamPassingKinds());
  auto sigIt = sigRange.begin();

  SmallVector<Type> newParamTypes;
  SmallVector<StringAttr> newParamNames;
  SmallVector<PassingKind> newPassingKinds;
  SmallVector<TypedAttr> newPosDefaults;
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
        // TODO: implement default kw-only struct parameters
        if (i >= defaultIdx)
          newPosDefaults.push_back(sig.getDefaultPosParams()[i - defaultIdx]);
        if (sig.isVarArg(i))
          paramVarArg = true;
      }
      ++sigIt;
      continue;
    }
    assert(cur == val && "cannot change bound parameter value");
  }
  assert(sigIt == sigRange.end() && "expected signature to get processed");

  // TODO: implement kw-only struct parameters
  auto newSig = TypeSignatureType::get(
      getContext(), newParamTypes, newParamNames, newPassingKinds,
      newPosDefaults, /*defaultKwOnlyParams=*/{}, paramVarArg);
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
  // Handle names, and index references.
  if (succeeded(p.parseOptionalStar())) {
    std::string str;
    // Resolve ambiguity with *"...".
    if (succeeded(p.parseOptionalString(&str))) {
      result = ParamDeclRefAttr::get(str, *this);
      return mlir::success();
    }

    // Try to parse *(0,0) as an index reference.
    if (succeeded(p.parseOptionalLParen())) {
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      bool isResult = succeeded(p.parseOptionalStar());
      result = ParamIndexRefAttr::get(depth, isResult, index, *this);
      return mlir::success();
    }

    // *[x,y] is an implicit lifetime ref.
    size_t depth, index;
    if (succeeded(p.parseOptionalLSquare())) {
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRSquare())
        return failure();
      result = ImplicitLifetimeRefAttr::get(depth, index, *this);
      return mlir::success();
    }
    // We don't support *?
    p.emitError(p.getCurrentLocation(), "unknown lifetime value");
    return failure();
  }

  // Handle unions as comma separated elements in braces.
  if (succeeded(p.parseOptionalLBrace())) {
    SmallVector<TypedAttr> elements;
    if (p.parseCommaSeparatedList(
            AsmParser::Delimiter::None,
            [&]() {
              elements.push_back({});
              return KGEN::parseParamValue(p, elements.back(), *this);
            },
            "in lifetime union") ||
        p.parseRBrace())
      return failure();
    result = LifetimeUnionAttr::get(elements, *this);
    return mlir::success();
  }

  // Handle mutability casts in parens.
  if (succeeded(p.parseOptionalLParen())) {
    TypedAttr operand;
    if (p.parseKeyword("mutcast") || parseLifetimeParamValue(p, operand) ||
        p.parseRParen())
      return failure();
    result = LifetimeMutCastAttr::get(operand, *this);
    return mlir::success();
  }
  return std::nullopt;
}

LogicalResult LifetimeType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto ref = ::dyn_cast<ImplicitLifetimeRefAttr>(value)) {
    p << "*[" << ref.getDepth() << ',' << ref.getIndex() << ']';
    return success();
  }

  if (auto unionAttr = ::dyn_cast<LifetimeUnionAttr>(value)) {
    // We know unions always have >1 element.
    p << "{";
    printParamValue(p, unionAttr.getOperand(0));
    for (auto operand : unionAttr.getOperands().drop_front()) {
      p << ", ";
      printParamValue(p, operand);
    }

    p << "}";
    return success();
  }

  if (auto mutcast = ::dyn_cast<LifetimeMutCastAttr>(value)) {
    p << "(mutcast ";
    printLifetimeParamValue(p, mutcast.getOperand());
    p << ")";
    return success();
  }

  return failure();
}

LifetimeType LifetimeType::get(TypedAttr isMutable) {
  assert(isMutable.getType().isSignlessInteger(1) &&
         "isMutable bit should be i1");
  return get(isMutable.getContext(), isMutable);
}

LifetimeType LifetimeType::get(MLIRContext *ctx, bool isMutable) {
  return get(ctx, BoolAttr::get(ctx, isMutable));
}

/// Return true if the mutable attribute is known to be the specific
/// constant.  This returns false if parametric or if the other value.
bool LifetimeType::isMutableKnown(bool value) {
  if (auto cst = ::dyn_cast<BoolAttr>(getIsMutable()))
    return cst.getValue() == value;
  return false;
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

RefType RefType::get(Type elementType, TypedAttr lifetime,
                     TypedAttr addrSpace) {
  assert(::isa<LifetimeType>(lifetime.getType()));
  return get(lifetime.getContext(), elementType, lifetime, addrSpace);
}

RefType RefType::get(Type elementType, TypedAttr lifetime, unsigned addrSpace) {
  auto *ctx = elementType.getContext();
  return get(elementType, lifetime,
             IntegerAttr::get(IndexType::get(ctx), addrSpace));
}

/// Return the pointer type that corresponds to this reference type, ignoring
/// the lifetime and the mutability.
PointerType RefType::getAsPointerType() {
  return PointerType::get(getElementType(), getAddressSpace());
}

/// Return this RefType but with a different element type.
RefType RefType::getWithElement(Type newElement) {
  return get(newElement, getLifetime(), getAddressSpace());
}

/// Return this RefType but with a different lifetime.
RefType RefType::getWithLifetime(TypedAttr newLifetime) {
  return get(getElementType(), newLifetime, getAddressSpace());
}

/// Return this RefType but with a different mutability.
RefType RefType::getWithMutability(bool isMut) {
  return get(getElementType(), LifetimeMutCastAttr::get(getLifetime(), isMut),
             getAddressSpace());
}

/// Return the type of the lifetime reference, which is always a
/// `!lit.lifetime<mutability>` type.
LifetimeType RefType::getLifetimeType() {
  return ::cast<LifetimeType>(getLifetime().getType());
}

/// Return a reference to the specified element type and mutability with an
/// immortal (#lit.lifetime) lifetime.
RefType RefType::getImmortal(Type elementType, bool isMut,
                             TypedAttr addrSpace) {
  return get(elementType, LifetimeAttr::get(elementType.getContext(), isMut),
             addrSpace);
}

RefType RefType::getImmortal(Type elementType, bool isMut, unsigned addrSpace) {
  return getImmortal(
      elementType, isMut,
      IntegerAttr::get(IndexType::get(elementType.getContext()), addrSpace));
}

/// Return true if the mutable attribute is known to be the specific
/// constant.  This returns false if parametric or if the other value.
bool RefType::isMutableKnown(bool value) {
  return ::cast<LifetimeType>(getLifetime().getType()).isMutableKnown(value);
}

/// Return a (possibly parameteric) specification for whether this reference
/// is a mutation or a read.
TypedAttr RefType::isMutable() {
  return ::cast<LifetimeType>(getLifetime().getType()).isMutable();
}

OptionalParseResult RefType::parseValue(AsmParser &p, TypedAttr &value) const {
  // Parse a `store_to_mem` directive.
  if (succeeded(p.parseOptionalKeyword("store_to_mem"))) {
    TypedAttr memValue;
    if (p.parseLParen() || parseParamValue(p, memValue, getElementType()) ||
        p.parseRParen())
      return failure();
    value = StoreToMemAttr::get(memValue, *this);
    return mlir::success();
  }

  return {};
}

LogicalResult RefType::printValue(AsmPrinter &p, TypedAttr value) const {
  // Print a `store_to_mem` directive.
  if (auto memAttr = ::dyn_cast<StoreToMemAttr>(value)) {
    p << "store_to_mem(";
    printParamValue(p, memAttr.getValue());
    p << ')';
    return success();
  }

  return failure();
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
  SmallVector<TypedAttr> defaultPosParams;
  SmallVector<TypedAttr> defaultKwOnlyParams;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  if (failed(parseOptionalParamSignature(
          p, inputParamTypes, resultParamTypes, paramNames, paramPassingKinds,
          defaultPosParams, defaultKwOnlyParams)))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaultPosArgs;
  SmallVector<TypedAttr> defaultKwOnlyArgs;
  SmallVector<ValueInputConvention> inputConventions;

  PassingKindParser passingKindParser(p);
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
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
    if (defaultVal) {
      if (passingKindParser.isCurrentKwOnly())
        defaultKwOnlyArgs.emplace_back(defaultVal);
      else
        defaultPosArgs.emplace_back(defaultVal);
    }

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
                          paramPassingKinds, defaultPosArgs, defaultPosParams,
                          defaultKwOnlyArgs, defaultKwOnlyParams,
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
      signature.getDefaultPosParams(), signature.getDefaultKwOnlyParams());

  ArrayRef<TypedAttr> defaultPosArgs = signature.getDefaultPosArgs();
  ArrayRef<PassingKind> argPassingKinds = signature.getArgPassingKinds();
  size_t numInputs = signature.getNumInputs();
  size_t defaultPosEnd = countNumPositional(argPassingKinds);
  size_t defaultPosStart = defaultPosEnd - defaultPosArgs.size();

  ArrayRef<TypedAttr> defaultKwOnlyArgs = signature.getDefaultKwOnlyArgs();
  size_t defaultKwOnlyEnd = numInputs - countNumImplicitKinds(argPassingKinds);
  size_t defaultKwOnlyStart = defaultKwOnlyEnd - defaultKwOnlyArgs.size();

  PassingKindPrinter passingKindPrinter(p, numInputs, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(argPassingKinds[i], i);

    StringAttr argName = signature.getArgName(i);
    if (!argName.empty()) {
      p.printString(argName);
      p << ": ";
    }

    p << signature.getValueInputs()[i];

    printInputConvention(p, signature.getInputConvention(i),
                         ValueInputConvention::OwnedInReg);

    if (i >= defaultPosStart && i < defaultPosEnd) {
      p << " = ";
      printParamValue(p, defaultPosArgs[i - defaultPosStart]);
    } else if (i >= defaultKwOnlyStart && i < defaultKwOnlyEnd) {
      p << " = ";
      printParamValue(p, defaultKwOnlyArgs[i - defaultKwOnlyStart]);
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

ArrayRef<TypedAttr> LITSignatureType::getDefaultPosArgs() {
  return getMetadata().getDefaultPosArgs();
}

ArrayRef<TypedAttr> LITSignatureType::getDefaultKwOnlyArgs() {
  return getMetadata().getDefaultKwOnlyArgs();
}

ArrayRef<TypedAttr> LITSignatureType::getDefaultPosParams() {
  return getMetadata().getDefaultPosParams();
}

ArrayRef<TypedAttr> LITSignatureType::getDefaultKwOnlyParams() {
  return getMetadata().getDefaultKwOnlyParams();
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
                          getDefaultPosArgs(), /*defaultPosParams=*/{},
                          getDefaultKwOnlyArgs(), /*defaultKwOnlyParams=*/{},
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

/// Get this signature with all the implicit lifetimes bound to #lit.lifetime
/// and dropped from the signature.
SignatureType LITSignatureType::getWithImplicitLifetimesBoundImmortal() {
  // Avoid work if this there is nothing to do.
  if (getNumImplicitLifetimeDecls() == 0)
    return *this;

  SmallVector<TypedAttr> lifetimes(getNumImplicitLifetimeDecls(),
                                   LifetimeAttr::get(getContext(), true));
  FunctionType newFnType = substituteImplicitLifetimesIntoValues(
      lifetimes,
      []() -> InFlightDiagnostic { llvm_unreachable("malformed fn type"); });

  return SignatureType::get(newFnType, getInputParamTypes(),
                            getResultParamTypes(), getInputConventions(),
                            getFnEffects());
}

/// This method replaces direct uses of NAMED implicit lifetime declarations
/// with index-based references.  lifetimeDecls specifies the names of the
/// implicit lifetime decls to replace.
///
/// If indexOffset is subtracted from depth when set.
Type LITSignatureType::replaceImplicitLifetimesWithIndexes(
    Type origType, ArrayRef<ParamDeclAttr> lifetimeDecls, size_t indexOffset) {

  // If there are no implicit lifetimes, then this is a noop.
  if (lifetimeDecls.empty())
    return origType;

  // Replace named implicit lifetime parameter references with index-based
  // references in the signature.
  struct LifetimeDeclRemapper : IndexParameterReplacer<LifetimeDeclRemapper> {
    LifetimeDeclRemapper(size_t indexOffset) : indexOffset(indexOffset) {}

    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = ::dyn_cast<ParamDeclRefAttr>(attr)) {
        if (auto it = mapping.find(ref.getName()); it != mapping.end()) {
          // Subtract indexOffset because we may be replacing the signature
          // directly.
          size_t index = it->second;
          return ImplicitLifetimeRefAttr::get(depth - indexOffset, index,
                                              ref.getType());
        }
      }
      return nullptr;
    }

    size_t indexOffset;
    DenseMap<StringAttr, size_t> mapping;
  } remapper(indexOffset);
  for (auto [i, decl] : llvm::enumerate(lifetimeDecls))
    remapper.mapping.try_emplace(decl.getName(), i);
  return remapper.replace(origType);
}

/// This method replaces direct uses of NAMED implicit lifetime declarations
/// with index-based references corresponding to the signature. `lifetimeDecls`
/// specifies the names of the implicit lifetime decls.
SignatureType LITSignatureType::replaceImplicitLifetimesWithIndexes(
    ArrayRef<ParamDeclAttr> lifetimeDecls) {
  assert(lifetimeDecls.size() == getNumImplicitLifetimeDecls() &&
         "Incorrect number of lifetime decls");
  return ::cast<SignatureType>(
      replaceImplicitLifetimesWithIndexes(*this, lifetimeDecls, 1));
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
    if (SignatureType::hasAddress(conv))
      ++result;
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
