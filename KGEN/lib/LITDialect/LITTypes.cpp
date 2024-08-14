//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
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
  dialect->registerMnemonicType<AnyStructType>();
  dialect->registerMnemonicType<TraitType>();
  dialect->registerMnemonicType<LifetimeType>();
  dialect->registerMnemonicType<LifetimeSetType>();

  // Register the StructType parser.
  getContext()->getLoadedDialect<KGENDialect>()->setSymbolTypeParser(
      [&](AsmParser &p, SymbolRefAttr symbol) -> FailureOr<Type> {
        SmallVector<TypedAttr> values;
        if (parseParameterValues(p, values))
          return failure();
        Type metatype;
        if (succeeded(p.parseOptionalColon())) {
          if (parseKGENType(p, metatype))
            return failure();
        } else {
          metatype = AnyStructType::get(symbol, values,
                                        TypeSignatureType::get(p.getContext()));
        }
        return StructType::get(symbol, values, metatype);
      });
}

//===----------------------------------------------------------------------===//
// TypeSignatureType
//===----------------------------------------------------------------------===//

/// TODO: remove these?
static ParseResult parseTypeSignature(AsmParser &p,
                                      SmallVectorImpl<Type> &paramTypes,
                                      PogListAttr &paramListAttrs) {
  if (parseOptionalParamSignature(p, paramTypes, paramListAttrs))
    return failure();
  return success();
}

static void printTypeSignature(AsmPrinter &p, ArrayRef<Type> paramTypes,
                               PogListAttr paramListAttrs) {
  printOptionalParamSignature(p, paramTypes, paramListAttrs);
}

LogicalResult
TypeSignatureType::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<Type> paramTypes,
                          PogListAttr paramListAttrs) {
  if (paramListAttrs.getPogs().size() != paramTypes.size()) {
    return emitError() << "number of parameters doesn't match number of input "
                          "parameter types";
  }

  return verifyDefaultTypes(emitError, paramListAttrs.getDefaultPos(),
                            paramListAttrs.getDefaultKwOnly(), paramListAttrs,
                            paramTypes, "parameter");
}

bool TypeSignatureType::isVarParam(size_t idx) const {
  return getParamListAttrs().isVariadic(idx);
}

bool TypeSignatureType::hasVariadicParam() const {
  return getParamListAttrs().hasVariadic();
}

TypeSignatureType TypeSignatureType::remapToSignature(
    function_ref<InFlightDiagnostic()> emitError, ParamDeclArrayAttr paramDecls,
    PogListAttr paramListAttrs) {
  IndexRefRemapper remapper(paramDecls, {});
  SmallVector<Type> inputParamTypes =
      llvm::map_to_vector(paramDecls, [&](ParamDeclAttr decl) {
        return remapper.replace(decl.getType());
      });

  MLIRContext *ctx = paramDecls.getContext();
  paramListAttrs = PogListAttr::get(
      ctx, paramListAttrs.getPogs(),
      remapper.replace(paramListAttrs.getDefaultPos()),
      remapper.replace(paramListAttrs.getDefaultKwOnly()),
      paramListAttrs.getPackIndex(), paramListAttrs.getOrigPackConvention());
  return TypeSignatureType::getChecked(emitError, ctx, inputParamTypes,
                                       paramListAttrs);
}

TypeSignatureType TypeSignatureType::get(MLIRContext *context) {
  return get(context, /*paramTypes=*/{}, PogListAttr::get(context));
}

StringAttr TypeSignatureType::getParamName(size_t idx) const {
  return getParamListAttrs().getName(idx);
}

ArrayRef<TypedAttr> TypeSignatureType::getDefaultPosParams() const {
  return getParamListAttrs().getDefaultPos();
}
ArrayRef<TypedAttr> TypeSignatureType::getDefaultKwOnlyParams() const {
  return getParamListAttrs().getDefaultKwOnly();
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

OptionalParseResult LIT::StructType::parseValue(AsmParser &p,
                                                TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return {};

  // Handle `{}`.
  if (succeeded(p.parseOptionalRBrace())) {
    value = LITStructAttr::get({}, *this);
    return mlir::success();
  }

  // Special-case `{<value>}`.
  std::string name;
  if (failed(p.parseOptionalKeywordOrString(&name))) {
    TypedAttr element;
    if (parseColonTypeParamValue(p, element))
      return failure();
    value = LITStructAttr::get(
        {{StringAttr::get(p.getContext(), "value"), element}}, *this);
    return p.parseRBrace();
  }

  // Parse `{(<name-type> = <value>)+}`.
  Type type;
  TypedAttr element;
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  auto parseElement = [&]() -> ParseResult {
    if (parseColonTypeOrIndex(p, type) || p.parseEqual() ||
        parseParamValue(p, element, type))
      return failure();
    values.emplace_back(StringAttr::get(p.getContext(), name), element);
    return success();
  };
  if (parseElement())
    return failure();
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseKeywordOrString(&name) || parseElement())
      return failure();
  }
  value = LITStructAttr::get(values, *this);

  return p.parseRBrace();
}

LogicalResult LIT::StructType::printValue(AsmPrinter &p,
                                          TypedAttr value) const {
  auto attr = ::dyn_cast<LITStructAttr>(value);
  if (!attr)
    return failure();
  ArrayRef<std::tuple<StringAttr, TypedAttr>> values = attr.getValues();

  p << '{';
  if (values.size() == 1 && std::get<0>(values.front()) == "value") {
    printColonTypeParamValue(p, std::get<1>(values.front()));
  } else {
    llvm::interleaveComma(values, p, [&](const auto &element) {
      auto [name, value] = element;
      p.printKeywordOrString(name);
      printColonTypeOrIndex(p, value.getType());
      p << " = ";
      printParamValue(p, value);
    });
  }
  p << '}';
  return success();
}

LIT::StructType LIT::StructType::get(SymbolRefAttr name,
                                     ArrayRef<TypedAttr> paramValues,
                                     Type metatype) {
  return get(name.getContext(), SymbolAttr::get(name), paramValues, metatype);
}

LIT::StructType LIT::StructType::get(SymbolRefAttr name, Type metatype) {
  return get(name, {}, metatype);
}

SymbolRefAttr LIT::StructType::getSymbol() const {
  return getValue().getValue();
}

std::optional<StringRef> LIT::StructType::getAliasName() {
  // Don't alias types with parameter references.
  if (!getParamValues().empty())
    return {};
  return getAliasName(getSymbol());
}

std::optional<StringRef> LIT::StructType::getAliasName(SymbolRefAttr symbol) {
  // Use the leaf name as the alias name.
  StringRef leaf = symbol.getLeafReference().getValue();
  unsigned offset = leaf.size();
  while (offset > 0 && std::isalnum(leaf[offset - 1]))
    --offset;
  if (offset == leaf.size() ||
      (!offset && symbol.getNestedReferences().empty()))
    return {};
  return leaf.substr(offset);
}

LogicalResult
LIT::StructType::verifySymbolUses(Operation *module,
                                  mlir::LockedSymbolTableCollection &symtab,
                                  Location loc) const {
  DeclInterface decl = ::dyn_cast_or_null<DeclInterface>(
      symtab.lookupSymbolIn(module, getSymbol()));
  if (!decl) {
    return mlir::emitError(loc)
           << getSymbol() << " does not reference a KGEN type declaration";
  }

  if (getParamValues().empty() && decl.getInputParams().empty())
    return success();

  // We have to specialize the type's parameter decls.
  ParameterEvaluator evaluator(decl.getInputParams(), getParamValues());
  SmallVector<ParamDeclAttr, 8> specializedDecls;
  for (ParamDeclAttr decl : decl.getInputParams())
    specializedDecls.push_back(
        ::cast<ParamDeclAttr>(evaluator.getReboundAttribute(decl)));

  return verifyParamDeclsMatch(
      "input parameter", "!lit.struct symbol use", getParamValues(), loc,
      getSymbol().getLeafReference(), specializedDecls, decl.getLoc());
}

static ParseResult
parseParameterizedSymbol(AsmParser &p, SymbolAttr &symbol,
                         SmallVectorImpl<TypedAttr> &paramValues) {
  if (p.parseCustomAttributeWithFallback(symbol) ||
      parseParameterValues(p, paramValues))
    return failure();
  return success();
}

static void printParameterizedSymbol(AsmPrinter &p, SymbolAttr symbol,
                                     ArrayRef<TypedAttr> paramValues) {
  if (!paramValues.empty() && succeeded(p.printAlias(symbol)))
    p << ' ';
  else
    p << symbol.getValue();
  printParameterValues(p, paramValues);
}

void LIT::StructType::printSymbol(AsmPrinter &p) const {
  // Use the alias printer if suitable.
  if (succeeded(p.printAlias(*this)))
    return;

  p << getSymbol();
  printParameterValues(p, getParamValues());
  if (auto mt = ::dyn_cast<AnyStructType>(getMetaType()))
    if (mt.getSignature().getInputParamTypes().empty())
      return;
  p << " : ";
  printKGENType(p, getMetaType());
}

static ParseResult parseOptionalMetaType(AsmParser &p, Type &metatype,
                                         SymbolAttr symbol,
                                         ArrayRef<TypedAttr> paramValues) {
  if (succeeded(p.parseOptionalComma()))
    return parseKGENType(p, metatype);

  metatype = AnyStructType::get(symbol.getValue(), paramValues,
                                TypeSignatureType::get(p.getContext()));
  return success();
}

static void printOptionalMetaType(AsmPrinter &p, Type metatype,
                                  SymbolAttr symbol,
                                  ArrayRef<TypedAttr> paramValues) {
  if (auto mt = dyn_cast<AnyStructType>(metatype))
    if (mt.getSignature().getInputParamTypes().empty())
      return;
  p << ", ";
  printKGENType(p, metatype);
}

/// Get the name of the referenced type, ignoring packages.
StringAttr LIT::StructType::getName() {
  auto symbol = getSymbol();
  if (symbol.getNestedReferences().empty())
    return symbol.getRootReference();
  return symbol.getNestedReferences().back().getAttr();
}

//===----------------------------------------------------------------------===//
// AnyStructType
//===----------------------------------------------------------------------===//

static OptionalParseResult parseTypeValue(AsmParser &p, TypedAttr &value,
                                          Type metatype) {
  auto typeParser = [metatype](AsmParser &p,
                               Type &typeValue) -> OptionalParseResult {
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
      typeValue = LIT::StructType::get(symbol, values, declRefMetaType);
    } else {
      result = parseOptionalKGENType(p, typeValue);
    }
    return result;
  };
  return parseSugaredTypeValue(p, value, metatype, typeParser);
}

static LogicalResult printTypeValue(AsmPrinter &p, TypedAttr value,
                                    Type metatype) {
  auto typePrinter = [metatype](AsmPrinter &p, Type type) {
    if (auto ref = ::dyn_cast<LIT::StructType>(type)) {
      // Use the alias printer if suitable.
      if (failed(p.printAlias(ref))) {
        p << ref.getSymbol();
        printParameterValues(p, ref.getParamValues());
        if (ref.getMetaType() != metatype) {
          p << " : ";
          printKGENType(p, ref.getMetaType());
        }
      }
    } else {
      printKGENType(p, type);
    }
  };
  return printSugaredTypeValue(p, value, typePrinter);
}

AnyStructType AnyStructType::get(SymbolRefAttr symbol,
                                 ArrayRef<TypedAttr> values,
                                 TypeSignatureType signature) {
  return get(symbol.getContext(), SymbolAttr::get(symbol), values, signature);
}

SymbolRefAttr AnyStructType::getSymbol() const { return getValue().getValue(); }

OptionalParseResult AnyStructType::parseValue(AsmParser &p,
                                              TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult AnyStructType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value, *this);
}

/// Return the struct type this metatype corresponds to.
LIT::StructType AnyStructType::getStructType() {
  return LIT::StructType::get(getSymbol(), getParamValues(), *this);
}

AnyStructType AnyStructType::bind(ArrayRef<TypedAttr> values) const {
  assert(getParamValues().size() == values.size() && "expected full value set");

  TypeSignatureType sig = getSignature();
  PogListAttr paramListAttr = sig.getParamListAttrs();
  auto sigRange = llvm::enumerate(sig.getParamTypes(), paramListAttr.getPogs());
  auto sigIt = sigRange.begin();

  SmallVector<Type> newParamTypes;
  SmallVector<PogMetadataAttr> newPogs;
  SmallVector<TypedAttr> newPosDefaults;
  SmallVector<TypedAttr> newKwOnlyDefaults;

  DefaultValueHandler defaultHandler(paramListAttr);
  ParameterEvaluator evaluator;
  for (auto [cur, val] : llvm::zip(getParamValues(), values)) {
    // Current value is unbound. This corresponds to a parameter in the
    // signature.
    if (::isa<UnboundAttr>(cur)) {
      if (::isa<UnboundAttr>(val)) {
        auto [i, type, pogAttr] = *sigIt;
        newParamTypes.push_back(evaluator.getReboundType(type));
        newPogs.push_back(pogAttr);

        if (TypedAttr defaultOr = defaultHandler.getPosDefault(i))
          newPosDefaults.push_back(defaultOr);
        else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(i))
          newKwOnlyDefaults.push_back(defaultOr);

        evaluator.addInputValue(ParamIndexRefAttr::get(newParamTypes.size() - 1,
                                                       newParamTypes.back()));
      } else {
        evaluator.addInputValue(val);
      }
      ++sigIt;
      continue;
    }
    assert(cur == val && "cannot change bound parameter value");
  }
  assert(sigIt == sigRange.end() && "expected signature to get processed");

  auto paramListAttrs = PogListAttr::get(getContext(), newPogs, newPosDefaults,
                                         newKwOnlyDefaults);
  auto newSig =
      TypeSignatureType::get(getContext(), newParamTypes, paramListAttrs);
  return AnyStructType::get(getSymbol(), values, newSig);
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

/// Return the metatype for this this trait as a value.
AnyTraitType TraitType::getMetaType() { return AnyTraitType::get(*this); }

/// Return a TypeConstantAttr for a reference to this trait as a value, e.g.
/// uttering 'Stringable' in code.
TypedAttr TraitType::getPValue() {
  return TypeConstantAttr::get(*this, getMetaType());
}

//===----------------------------------------------------------------------===//
// AnyTraitType
//===----------------------------------------------------------------------===//

OptionalParseResult AnyTraitType::parseValue(AsmParser &p,
                                             TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult AnyTraitType::printValue(AsmPrinter &p, TypedAttr value) const {
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

/// Classify the mutability into Mutable/Immutable/Parametric.
LifetimeType::MutabilityClass LifetimeType::getMutabilityClass() {
  auto cst = ::dyn_cast<BoolAttr>(getIsMutable());
  if (!cst)
    return Parametric;
  return cst.getValue() ? Mutable : Immutable;
}

//===----------------------------------------------------------------------===//
// LifetimeSetType
//===----------------------------------------------------------------------===//

OptionalParseResult LifetimeSetType::parseValue(AsmParser &p,
                                                TypedAttr &value) const {
  SmallVector<TypedAttr> lifetimes;
  OptionalParseResult result = parseOptionalLifetimeSet(p, lifetimes);
  if (result.has_value()) {
    if (failed(*result))
      return failure();
    value = LifetimeSetAttr::get(getContext(), lifetimes, *this);
    return mlir::success();
  }
  return std::nullopt;
}

LogicalResult LifetimeSetType::printValue(AsmPrinter &p,
                                          TypedAttr value) const {
  if (auto set = ::dyn_cast<LifetimeSetAttr>(value)) {
    printLifetimeSet(p, set.getOperands());
    return success();
  }
  return failure();
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

/// Classify the mutability into Mutable/Immutable/Parametric.
LifetimeType::MutabilityClass RefType::getMutabilityClass() {
  return ::cast<LifetimeType>(getLifetime().getType()).getMutabilityClass();
}

/// Return a (possibly parameteric) specification for whether this reference
/// is a mutation or a read.
TypedAttr RefType::isMutable() {
  return ::cast<LifetimeType>(getLifetime().getType()).isMutable();
}

/// Return true if this is in address space 0.
bool RefType::isDefaultAddrSpace() {
  if (auto intAttr = ::dyn_cast<IntegerAttr>(getAddressSpace()))
    return intAttr.getInt() == 0;
  return false;
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
// RefPackType
//===----------------------------------------------------------------------===//

RefPackType RefPackType::get(TypedAttr variadic, TypedAttr lifetime,
                             TypedAttr addressSpace) {
  return get(variadic.getContext(), variadic, lifetime, addressSpace);
}

VariadicAttr RefPackType::getVariadicIfResolved() const {
  return ::dyn_cast<VariadicAttr>(getVariadic());
}

/// Return the effective type (always a reference) of each element given
/// the type according to the type list.
RefType RefPackType::getElementRefTypeFor(Type elementType) {
  return RefType::get(elementType, getLifetime(), getAddressSpace());
}

/// This returns the element type of the variadic list parameter, typically
/// something like !kgen.type or a trait type.
Type RefPackType::getVariadicElementType() {
  return ::cast<VariadicType>(getVariadic().getType()).getElementType();
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

  SmallVector<Type> inputParamTypes;
  PogListAttr paramListAttr;
  if (failed(parseOptionalParamSignature(p, inputParamTypes, paramListAttr)))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaultPosArgs;
  SmallVector<TypedAttr> defaultKwOnlyArgs;
  SmallVector<ArgConvention> argConventions;
  SmallVector<size_t> argVariadicIndices;
  ssize_t argPackIndex = -1;
  std::optional<ArgConvention> origArgPackConvention;

  PassingKindParser passingKindParser(p);
  size_t idx = 0;
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
        parseConventionAndVariadicness(p, argConventions.emplace_back(),
                                       argVariadicIndices, argPackIndex,
                                       origArgPackConvention, idx++))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(
            p, defaultVal, type,
            SignatureType::hasAddress(argConventions.back()))))
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
  auto metadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, argNames, argPassingKinds, defaultPosArgs,
                       defaultKwOnlyArgs, argVariadicIndices, argPackIndex,
                       origArgPackConvention),
      paramListAttr, numLifetimeDecls);
  signature = SignatureType::getChecked(
      [&] { return p.emitError(startLoc); }, functionType, inputParamTypes,
      /*resultParamTypes=*/{}, argConventions, effects, metadata);
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

  printOptionalParamSignature(p, signature.getParamTypes(),
                              signature.getParamListAttrs());

  PogListAttr argListAttr = signature.getArgListAttrs();
  SmallVector<Variadicness> variadicness = getVariadicness(argListAttr);
  DefaultValueHandler defaultHandler(argListAttr);
  PassingKindPrinter passingKindPrinter(p, argListAttr, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(i);

    StringAttr argName = signature.getArgName(i);
    if (!argName.empty()) {
      p.printString(argName);
      p << ": ";
    }

    p << signature.getArguments()[i];
    ArgConvention argConv = signature.getArgConvention(i);
    if (variadicness[i] == Variadicness::kPack) {
      assert(argConv == ArgConvention::BorrowedInReg ||
             argConv == ArgConvention::OwnedInReg);
      argConv = signature.getPackVarArgConvention(i);
    }
    printConventionAndVariadicness(p, argConv, variadicness[i]);

    if (TypedAttr defaultOr = defaultHandler.getDefault(i)) {
      p << " = ";
      printParamValue(p, defaultOr);
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

PogListAttr LITSignatureType::getArgListAttrs() {
  return getMetadata().getArgListAttrs();
}

PogListAttr LITSignatureType::getParamListAttrs() {
  return getMetadata().getParamListAttrs();
}

StringAttr LITSignatureType::getArgName(size_t idx) {
  return getArgListAttrs().getName(idx);
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

StringAttr LITSignatureType::getParamName(size_t idx) {
  return getParamListAttrs().getName(idx);
}

/// Get the number of implicit lifetime decls this function type carries.
size_t LITSignatureType::getNumImplicitLifetimeDecls() {
  return getMetadata().getNumImplicitLifetimeDecls();
}

Type LITSignatureType::getUserResultType() {
  return LIT::getSignatureUserResultType(*this, getArguments(),
                                         getResultType());
}

LITSignatureType LITSignatureType::dropParamValues() {
  return get(
      getValues(), /*paramTypes=*/{}, getArgConventions(), getFnEffects(),
      FnMetadataAttr::get(getArgListAttrs(), /*numImplicitLifetimeDecls=*/0));
}

bool LITSignatureType::isAnyVarArg(size_t index) {
  return getMetadata().isAnyVarArg(index);
}

bool LITSignatureType::isPosVarArg(size_t index) {
  return getMetadata().isPosVarArg(index);
}

/// For a PosVarArg, return the declared ArgConvention of the elements. For
/// example: fn x(inout *args: Int) is declared 'inout'.
ArgConvention LITSignatureType::getPosVarArgConvention(size_t index) {
  assert(isPosVarArg(index) && "isn't a positional vararg");
  return ::cast<VariadicType>(getArguments()[index]).getConvention();
}

bool LITSignatureType::isKwVarArg(size_t index) {
  return getMetadata().isKwVarArg(index);
}

bool LITSignatureType::isPackVarArg(size_t index) {
  return getMetadata().isPackVarArg(index);
}

/// If the specified argument is a variadic pack, return the VariadicPack.
Type LITSignatureType::getIfVariadicPack(size_t index) {
  if (!isPackVarArg(index))
    return {};
  return getArguments()[index];
}

/// For a PosVarArg, return the declared ArgConvention of the elements. For
/// example: fn x(inout *args: Int) is declared 'inout'.
ArgConvention LITSignatureType::getPackVarArgConvention(size_t index) {
  assert(getMetadata().isPackVarArg(index));
  return *getArgListAttrs().getOrigPackConvention();
}

bool LITSignatureType::isParamVarArg(size_t index) {
  return getParamListAttrs().isVariadic(index);
}

bool LITSignatureType::hasParamVarArgs() {
  return getMetadata().hasParamVarArgs();
}

bool LITSignatureType::hasPackVarArgs() {
  return getMetadata().hasPackVarArgs();
}

bool LITSignatureType::hasKwVarArgs() { return getMetadata().hasKwVarArgs(); }

unsigned LITSignatureType::getErrorSlotOffset() {
  assert(isThrows() && "signature does not refer to a throwing function");
  return 1 + hasMemoryOnlyResult();
}

/// Substitute the specified implicit lifetime references into the specified
/// type, replacing them with `values` if they are at depth 0, or decrementing
/// their depth if not.  This returns the resultant FunctionType on success,
/// and invokes 'emitError'+returns null on error.
FunctionType LITSignatureType::substituteImplicitLifetimesIntoValues(
    ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError) {

  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = ::dyn_cast<ImplicitLifetimeRefAttr>(attr);
          ref && ref.getDepth() == depth) {
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
  FunctionType result = substitutor.replace(getValues());
  return substitutor.hadError ? FunctionType() : result;
}

/// Get this signature with all the implicit lifetimes bound to #lit.lifetime
/// and dropped from the signature.
LITSignatureType LITSignatureType::getWithImplicitLifetimesBoundImmortal() {
  // Avoid work if there is nothing to do.
  if (getNumImplicitLifetimeDecls() == 0)
    return *this;

  // Replace the lifetimes with attrs of the right mutability.  We just scan
  // through the type to find the references to update.  We get implicit
  // lifetimes in a range of places (e.g. buried in pack and variadic types etc)
  // that make it difficult to "just know" the mutability of each one.
  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // If we are substituting the signature directly, subtract 1.
      auto ref = ::dyn_cast<ImplicitLifetimeRefAttr>(attr);
      if (!ref || ref.getDepth() != depth)
        return nullptr;
      return LifetimeAttr::get(ref.getType());
    }
  };

  FunctionType newFnType = Substitutor().replace(getValues());
  return LITSignatureType::get(newFnType, getParamTypes(), getArgConventions(),
                               getFnEffects(), getMetadata());
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
LITSignatureType LITSignatureType::replaceImplicitLifetimesWithIndexes(
    ArrayRef<ParamDeclAttr> lifetimeDecls) {
  assert(lifetimeDecls.size() == getNumImplicitLifetimeDecls() &&
         "Incorrect number of lifetime decls");
  return ::cast<LITSignatureType>(
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
size_t LITSignatureType::countImplicitLifetimes(ArrayRef<ArgConvention> convs) {
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
  SmallVector<PogMetadataAttr> argPogs(
      numInputs,
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  auto metadata = FnMetadataAttr::get(PogListAttr::get(ctx, argPogs),
                                      numImplicitLifetimeDecls);
  return LITSignatureType::get(funcType, /*paramTypes=*/{},
                               /*convs=*/{}, /*effects=*/{}, metadata);
}

LITSignatureType LITSignatureType::get(FunctionType values,
                                       ArrayRef<Type> paramTypes,
                                       ArrayRef<ArgConvention> convs,
                                       FnEffects effects,
                                       FnMetadataAttr metadata) {
  assert(metadata && "LITSignatureType must have non-null metadata");
  return SignatureType::get(values, paramTypes, /*resultParamTypes=*/{}, convs,
                            effects, metadata);
}

/// Reconstruct the signature using a list of named input parameters and
/// indices indicating which one of them are variadic. These parameters are
/// prepended to the current signature and references are remapped to index
/// references. An additional array of indices corresponding to variadic
/// parameters of the prepended parameters is also required.
LITSignatureType
LITSignatureType::prependParams(LITSignatureType sig,
                                ArrayRef<ParamDeclAttr> parentParams,
                                ArrayRef<bool> parentVariadicMask) {
  IndexRefRemapper remapper(parentParams, /*resultParams=*/{},
                            parentParams.size());
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : sig.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  FnMetadataAttrInterface metadata =
      remapper.replace(sig.getMetadata().prependPosParams(parentParams.size(),
                                                          parentVariadicMask));

  return SignatureType::get(remapper.replace(sig.getValues()), inputParamTypes,
                            remapper.replace(sig.getResultParamTypes()),
                            sig.getArgConventions(), sig.getFnEffects(),
                            metadata);
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

Type LIT::getSignatureUserResultType(SignatureType sigType,
                                     ArrayRef<Type> argTypes, Type resultType) {
  // If this function has an init_self argument, then it returns None.
  if (sigType.hasInitSelfArg())
    return KGEN::NoneType::get(sigType.getContext());
  // If this function has a byref_result, return the reference element type.
  if (sigType.hasMemoryOnlyResult())
    return cast<RefType>(argTypes.back()).getElementType();
  return resultType;
}

/// The Lit parser and KGEN have different semantics for binding function
/// argument and result types. The parser will evaluate 'apply' expressions, but
/// KGEN does not since it cannot always have access to a symbol table.
/// Specialize a signature type while rebinding the input parameter values to
/// the expected input parameter types.
std::pair<LITSignatureType, ParameterExprArrayAttr>
LIT::getUnboundSpecializedSignature(LITSignatureType type,
                                    ParameterExprArrayAttr bindings) {
  if (bindings.empty())
    return {type, bindings};

  // KGEN expects different bindings types than Lit can provide. Rebind the
  // parameters to the expected types.
  SmallVector<TypedAttr> unboundBindings;
  ParameterEvaluator evaluator;
  for (auto [binding, type] : llvm::zip(bindings, type.getParamTypes())) {
    TypedAttr value = binding;
    Type unboundType = evaluator.getReboundType(type);
    if (unboundType != value.getType())
      value = ParamOperatorAttr::get(POC::Rebind, value, unboundType);
    evaluator.addInputValue(value);
    unboundBindings.push_back(value);
  }
  type = type.getSpecializedSignature(
      unboundBindings, [&]() -> InFlightDiagnostic {
        return mlir::emitError(UnknownLoc::get(type.getContext()));
      });
  assert(type && "bad bindings specified");
  return {type,
          ParameterExprArrayAttr::get(type.getContext(), unboundBindings)};
}

/// This returns the singleton value to use for a parameter value that
/// `isSingletonParameter` returns true on. This aborts on non-singleton types.
TypedAttr LIT::getSingletonParameterValue(Type type) {
  // TODO: Could support structs of lifetimes.
  if (auto lifetime = dyn_cast<LifetimeType>(type))
    return LifetimeAttr::get(lifetime);
  llvm_unreachable("isn't a singleton parameter");
}
