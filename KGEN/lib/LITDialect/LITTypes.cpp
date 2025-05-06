//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
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
  dialect->registerMnemonicType<MetaType>();
  dialect->registerMnemonicType<TraitType>();
  dialect->registerMnemonicType<OriginType>();
  dialect->registerMnemonicType<OriginSetType>();

  // Register the StructType parser.
  getContext()->getLoadedDialect<KGENDialect>()->setSymbolTypeParser(
      [&](AsmParser &p, SymbolRefAttr symbol) -> FailureOr<Type> {
        SmallVector<TypedAttr> values;
        if (parseParameterValues(p, values))
          return failure();
        // Delete all this meta type stuff and remove from tests.
        TypeSignatureType typeSig;
        if (succeeded(p.parseOptionalColon())) {
          Type metatype;
          if (parseKGENType(p, metatype))
            return failure();
          if (auto anyStruct = dyn_cast<StructMetaType>(metatype))
            typeSig = anyStruct.getSignature();
        }
        if (!typeSig)
          typeSig = TypeSignatureType::get(p.getContext());
        return StructType::get(symbol, values, typeSig);
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
  if (paramListAttrs.size() != paramTypes.size()) {
    return emitError() << "number of parameters doesn't match number of input "
                          "parameter types";
  }

  return verifyDefaultTypes(emitError, paramListAttrs.getDefaultPos(),
                            paramListAttrs.getDefaultKwOnly(), paramListAttrs,
                            paramTypes, "parameter");
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
  paramListAttrs =
      PogListAttr::get(ctx, paramListAttrs.getPogs(),
                       remapper.replace(paramListAttrs.getDefaultPos()),
                       remapper.replace(paramListAttrs.getDefaultKwOnly()),
                       paramListAttrs.getOrigPackConvention());
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

/// Bind parameter values to the signature, returning a new one.
TypeSignatureType TypeSignatureType::bind(ArrayRef<TypedAttr> values) const {
  assert(values.size() == getParamTypes().size() &&
         "expected full value set with UnboundAttrs for missing ones");

  PogListAttr paramListAttr = getParamListAttrs();

  SmallVector<Type> newParamTypes;
  SmallVector<PogMetadataAttr> newPogs;
  SmallVector<TypedAttr> newPosDefaults;
  SmallVector<TypedAttr> newKwOnlyDefaults;

  DefaultValueHandler defaultHandler(paramListAttr);
  ParameterEvaluator evaluator;
  for (auto [i, val, type, pogAttr] :
       llvm::enumerate(values, getParamTypes(), paramListAttr.getPogs())) {
    // If the current value is bound and we have a specified value, use it.
    if (!::isa<UnboundAttr>(val)) {
      evaluator.addInputValue(val);
      continue;
    }

    // Otherwise it is still unbound, maintain it as such.
    newParamTypes.push_back(evaluator.getReboundType(type));
    newPogs.push_back(pogAttr);

    if (TypedAttr defaultOr = defaultHandler.getPosDefault(i))
      newPosDefaults.push_back(evaluator.replace(defaultOr));
    else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(i))
      newKwOnlyDefaults.push_back(evaluator.replace(defaultOr));

    evaluator.addInputValue(
        ParamIndexRefAttr::get(newParamTypes.size() - 1, newParamTypes.back()));
  }
  auto paramListAttrs = PogListAttr::get(getContext(), newPogs, newPosDefaults,
                                         newKwOnlyDefaults);
  return TypeSignatureType::get(getContext(), newParamTypes, paramListAttrs);
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
  if (values.size() == 1 && std::get<0>(values.front()) == "value" &&
      // Don't print 'add(x, y)' as the value, because the parser will think
      // that is a field name.
      !::isa<ParamOperatorAttr>(std::get<1>(values.front()))) {
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
                                     TypeSignatureType signature) {
  return get(name.getContext(), SymbolAttr::get(name), paramValues, signature);
}

LIT::StructType LIT::StructType::get(SymbolRefAttr name,
                                     TypeSignatureType signature) {
  return get(name, {}, signature);
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
LIT::StructType::verifySymbolUses(SymTabEvaluationContext &evaluationContext,
                                  Location loc) const {
  Operation *module = evaluationContext.module;
  mlir::LockedSymbolTableCollection &symtab = evaluationContext.symtab;

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
  evaluator.setEvaluationContext(&evaluationContext);
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
}

/// Get the name of the referenced type, ignoring packages.
StringAttr LIT::StructType::getName() {
  auto symbol = getSymbol();
  if (symbol.getNestedReferences().empty())
    return symbol.getRootReference();
  return symbol.getNestedReferences().back().getAttr();
}

LIT::StructType LIT::StructType::bindAll(ArrayRef<TypedAttr> values) const {
  assert(getParamValues().size() == values.size() && "expected full value set");

  // The AnyStruct will have all of the parameters specified, e.g. something
  // like:
  // StructMetaType[Int : AnyType, UnboundAttr : I8, 42 : Int, UnboundAttr: F32]
  // but the TypeSignatureType will just have [I8, F32].  The input value
  // bindings must line up where they are already specified, but can further
  // refine the SignatureType.  See what to pass down to it.

  SmallVector<TypedAttr> newSignatureBindings;
  bool hadNewBinding = false;
  for (auto [cur, val] : llvm::zip(getParamValues(), values)) {
    // If the current value is bound, maintain it.
    if (!::isa<UnboundAttr>(cur)) {
      assert(cur == val && "cannot change bound parameter value");
    } else {
      hadNewBinding |= !::isa<UnboundAttr>(val);
      // Otherwise, propagate it into the TypeSignatureType.
      newSignatureBindings.push_back(val);
    }
  }

  // If we're refining our signature because we have new bindings, return an
  // AnyStruct with the updated signature and values.
  if (!hadNewBinding)
    return *this;

  auto newSig = getSignature().bind(newSignatureBindings);
  return LIT::StructType::get(getSymbol(), values, newSig);
}

LIT::StructType LIT::StructType::bindUnbound(ArrayRef<TypedAttr> values) const {
  SmallVector<TypedAttr> bindings;
  auto it = values.begin();
  for (TypedAttr value : getParamValues()) {
    if (::isa<UnboundAttr>(value))
      bindings.push_back(*it++);
    else
      bindings.push_back(value);
  }
  assert(it == values.end() && "expected all bindings to be consumed");
  return bindAll(bindings);
}

//===----------------------------------------------------------------------===//
// StructMetaType
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
      TypeSignatureType typeSig;
      if (auto anyStruct = dyn_cast<StructMetaType>(metatype))
        typeSig = anyStruct.getSignature();
      else
        typeSig = TypeSignatureType::get(p.getContext());
      typeValue = LIT::StructType::get(symbol, values, typeSig);
    } else {
      result = parseOptionalKGENType(p, typeValue);
    }
    return result;
  };
  return parseSugaredTypeValue(p, value, metatype, typeParser);
}

static LogicalResult printTypeValue(AsmPrinter &p, TypedAttr value) {
  auto typePrinter = [](AsmPrinter &p, Type type) {
    if (auto ref = ::dyn_cast<LIT::StructType>(type)) {
      // Use the alias printer if suitable.
      if (failed(p.printAlias(ref))) {
        p << ref.getSymbol();
        printParameterValues(p, ref.getParamValues());
      }
    } else {
      printKGENType(p, type);
    }
  };
  return printSugaredTypeValue(p, value, typePrinter);
}

//===----------------------------------------------------------------------===//
// TraitType
//===----------------------------------------------------------------------===//

OptionalParseResult TraitType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult TraitType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value);
}

/// Return the metatype for this this trait as a value.
AnyTraitType TraitType::getMetaType() { return AnyTraitType::get(*this); }

/// Return a TypeParamAttr for a reference to this trait as a value, e.g.
/// uttering 'Stringable' in code.
TypedAttr TraitType::getPValue() {
  return TypeParamAttr::get(*this, getMetaType());
}

//===----------------------------------------------------------------------===//
// AnyTraitType
//===----------------------------------------------------------------------===//

OptionalParseResult AnyTraitType::parseValue(AsmParser &p,
                                             TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult AnyTraitType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value);
}

//===----------------------------------------------------------------------===//
// MetaType
//===----------------------------------------------------------------------===//

OptionalParseResult MetaType::parseValue(AsmParser &p, TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult MetaType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value);
}

//===----------------------------------------------------------------------===//
// OriginType
//===----------------------------------------------------------------------===//

OptionalParseResult OriginType::parseValue(AsmParser &p,
                                           TypedAttr &result) const {
  // If there are any postfix origin syntax (<whatever>.field1.field2), then
  // parse them into 'result'.
  auto processPostFix = [&]() -> OptionalParseResult {
    if (!result)
      return failure();
    while (true) {
      if (succeeded(p.parseOptionalArrow())) {
        StringRef fieldName;
        if (failed(p.parseKeyword(&fieldName)))
          return failure();
        result = OriginFieldAttr::get(
            result, StringAttr::get(p.getContext(), fieldName));
        continue;
      }
      if (succeeded(p.parseOptionalLSquare())) {
        if (p.parseRSquare())
          return failure();
        result = IndirectOriginAttr::get(result);
        continue;
      }
      // Otherwise, not a postfix thing.
      break;
    }
    return mlir::success();
  };

  // Parse |...| as OriginSet and OriginSetUnion.
  if (succeeded(p.parseOptionalVerticalBar())) {
    TypedAttr set;
    if (parseParamValue(p, set, OriginSetType::get(p.getContext())) ||
        p.parseVerticalBar())
      return failure();
    result = OriginSetUnionAttr::get(set, *this);
    return mlir::success();
  }

  // Handle names, and index references.
  if (succeeded(p.parseOptionalStar())) {
    std::string str;
    // Resolve ambiguity with *"...".
    if (succeeded(p.parseOptionalString(&str))) {
      result = ParamDeclRefAttr::get(str, *this);
      return processPostFix();
    }

    // Try to parse *(0,0) as an index reference.
    if (succeeded(p.parseOptionalLParen())) {
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      result = ParamIndexRefAttr::get(depth, index, *this);
      return processPostFix();
    }

    // *[x,y] is an ImplicitOriginRefAttr.
    size_t depth, index;
    if (succeeded(p.parseOptionalLSquare())) {
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRSquare())
        return failure();
      result = ImplicitOriginRefAttr::get(depth, index, *this);
      return processPostFix();
    }

    // We don't support *?
    p.emitError(p.getCurrentLocation(), "unknown origin value");
    return failure();
  }

  // Handle unions as comma separated elements in braces.
  if (succeeded(p.parseOptionalLBrace())) {
    SmallVector<TypedAttr> elements;
    // Body is {} or {elts}
    if (failed(p.parseOptionalRBrace())) {
      if (p.parseCommaSeparatedList(
              AsmParser::Delimiter::None,
              [&]() {
                elements.push_back({});
                return KGEN::parseParamValue(p, elements.back(), *this);
              },
              "in origin union") ||
          p.parseRBrace())
        return failure();
    }
    result = OriginUnionAttr::get(elements, *this);
    return processPostFix();
  }

  // Handle mutability casts in parens.
  if (succeeded(p.parseOptionalLParen())) {
    TypedAttr operand;
    if (p.parseKeyword("mutcast") || parseOriginParamValue(p, operand) ||
        p.parseRParen())
      return failure();
    result = OriginMutCastAttr::get(operand, *this);
    return processPostFix();
  }

  // If this is a '*'-prefixed double quoted string, then this is a simple
  // parameter reference.
  if (succeeded(p.parseOptionalStar())) {
    if (succeeded(p.parseOptionalLParen())) {
      // Try to parse *(0,0) as an index reference.
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      result = ParamIndexRefAttr::get(depth, index, *this);
    } else {
      std::string name;
      if (failed(p.parseString(&name)))
        return failure();
      result = ParamDeclRefAttr::get(name, *this);
    }
    return processPostFix();
  }

  // Barewords / MLIR keywords are implicitly parameter declaration references
  // or the start of a expression in function form.
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // A bareword or string must be a parameter reference.
    result = ParamDeclRefAttr::get(keyword, *this);
    return processPostFix();
  }

  return {};
}

LogicalResult OriginType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto declRef = ::dyn_cast<ParamDeclRefAttr>(value)) {
    printParamName(p, declRef.getName(), /*isRef*/ false);
    return success();
  }

  if (auto set = ::dyn_cast<OriginSetUnionAttr>(value)) {
    p << '|';
    printParamValue(p, set.getValue());
    p << '|';
    return success();
  }

  if (auto ref = ::dyn_cast<ImplicitOriginRefAttr>(value)) {
    p << "*[" << ref.getDepth() << ',' << ref.getIndex() << ']';
    return success();
  }

  if (auto unionAttr = ::dyn_cast<OriginUnionAttr>(value)) {
    p << '{';
    if (unionAttr.getNumOperands()) {
      printParamValue(p, unionAttr.getOperand(0));
      for (auto operand : unionAttr.getOperands().drop_front()) {
        p << ", ";
        printParamValue(p, operand);
      }
    }
    p << '}';
    return success();
  }

  if (auto mutcast = ::dyn_cast<OriginMutCastAttr>(value)) {
    p << "(mutcast ";
    printOriginParamValue(p, mutcast.getOperand());
    p << ")";
    return success();
  }

  // Print field access with dot notation.
  if (auto field = ::dyn_cast<OriginFieldAttr>(value)) {
    if (failed(printValue(p, field.getBase())))
      return failure();
    // FIXME: This should use ".field" instead of "->field" but MLIR doesn't
    // make it easy to parse a dot.
    p << "->";
    printParamName(p, field.getField(), /*isRef*/ false);
    return success();
  }

  // Print field access with x.y[] notation.
  if (auto indirect = ::dyn_cast<IndirectOriginAttr>(value)) {
    if (failed(printValue(p, indirect.getBase())))
      return failure();
    p << "[]";
    return success();
  }

  return failure();
}

OriginType OriginType::get(TypedAttr isMutable) {
  assert(isMutable.getType().isSignlessInteger(1) &&
         "isMutable bit should be i1");
  return get(isMutable.getContext(), isMutable);
}

OriginType OriginType::get(MLIRContext *ctx, bool isMutable) {
  return get(ctx, BoolAttr::get(ctx, isMutable));
}

/// Return true if the mutable attribute is known to be the specific
/// constant.  This returns false if parametric or if the other value.
bool OriginType::isMutableKnown(bool value) {
  if (auto cst = ::dyn_cast<BoolAttr>(getIsMutable()))
    return cst.getValue() == value;
  return false;
}

/// Classify the mutability into Mutable/Immutable/Parametric.
OriginType::MutabilityClass OriginType::getMutabilityClass() {
  auto cst = ::dyn_cast<BoolAttr>(getIsMutable());
  if (!cst)
    return Parametric;
  return cst.getValue() ? Mutable : Immutable;
}

/// Remove any OriginMutCast and ._mlir_origin if present.
TypedAttr OriginType::stripMutCastAndFieldExtract(TypedAttr origin) {
  // Handle an extract out of an Origin type.
  if (auto extract = ::dyn_cast<LIT::StructExtractAttr>(origin)) {
    if (extract.getField() == ORIGIN_FIELD_NAME)
      return stripMutCastAndFieldExtract(extract.getStructValue());
  }

  // Ignore MutCasts.
  if (auto mutCast = ::dyn_cast<OriginMutCastAttr>(origin))
    return stripMutCastAndFieldExtract(mutCast.getOperand());

  return origin;
}

//===----------------------------------------------------------------------===//
// OriginSetType
//===----------------------------------------------------------------------===//

OptionalParseResult OriginSetType::parseValue(AsmParser &p,
                                              TypedAttr &value) const {
  SmallVector<TypedAttr> origins;
  OptionalParseResult result = parseOptionalOriginSet(p, origins);
  if (result.has_value()) {
    if (failed(*result))
      return failure();
    value = OriginSetAttr::get(getContext(), origins, *this);
    return mlir::success();
  }
  return std::nullopt;
}

LogicalResult OriginSetType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto set = ::dyn_cast<OriginSetAttr>(value)) {
    printOriginSet(p, set.getOperands());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

RefType RefType::get(Type elementType, TypedAttr origin, TypedAttr addrSpace) {
  assert(::isa<OriginType>(origin.getType()));
  return get(origin.getContext(), elementType, origin, addrSpace);
}

RefType RefType::get(Type elementType, TypedAttr origin, unsigned addrSpace) {
  auto *ctx = elementType.getContext();
  return get(elementType, origin,
             IntegerAttr::get(IndexType::get(ctx), addrSpace));
}

/// Return the pointer type that corresponds to this reference type, ignoring
/// the origin and the mutability.
PointerType RefType::getAsPointerType() {
  return PointerType::get(getElementType(), getAddressSpace());
}

/// Return this RefType but with a different element type.
RefType RefType::getWithElement(Type newElement) {
  return get(newElement, getOrigin(), getAddressSpace());
}

/// Return this RefType but with a different origin.
RefType RefType::getWithOrigin(TypedAttr newOrigin) {
  return get(getElementType(), newOrigin, getAddressSpace());
}

/// Return this RefType but with a different mutability.
RefType RefType::getWithMutability(bool isMut) {
  return get(getElementType(), OriginMutCastAttr::get(getOrigin(), isMut),
             getAddressSpace());
}

/// Return the type of the origin reference, which is always a
/// `!lit.origin<mutability>` type.
OriginType RefType::getOriginType() {
  return ::cast<OriginType>(getOrigin().getType());
}

/// Return a reference to the specified element type and mutability with
/// #lit.any.origin.
RefType RefType::getAnyOrigin(Type elementType, bool isMut,
                              TypedAttr addrSpace) {
  return get(elementType, AnyOriginAttr::get(elementType.getContext(), isMut),
             addrSpace);
}

RefType RefType::getAnyOrigin(Type elementType, bool isMut,
                              unsigned addrSpace) {
  return getAnyOrigin(
      elementType, isMut,
      IntegerAttr::get(IndexType::get(elementType.getContext()), addrSpace));
}

/// Return true if the mutable attribute is known to be the specific
/// constant.  This returns false if parametric or if the other value.
bool RefType::isMutableKnown(bool value) {
  return ::cast<OriginType>(getOrigin().getType()).isMutableKnown(value);
}

/// Classify the mutability into Mutable/Immutable/Parametric.
OriginType::MutabilityClass RefType::getMutabilityClass() {
  return ::cast<OriginType>(getOrigin().getType()).getMutabilityClass();
}

/// Return a (possibly parameteric) specification for whether this reference
/// is a mutation or a read.
TypedAttr RefType::isMutable() {
  return ::cast<OriginType>(getOrigin().getType()).isMutable();
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

RefPackType RefPackType::get(TypedAttr variadic, TypedAttr origin,
                             TypedAttr addressSpace) {
  return get(variadic.getContext(), variadic, origin, addressSpace);
}

VariadicAttr RefPackType::getVariadicIfResolved() const {
  return ::dyn_cast<VariadicAttr>(getVariadic());
}

/// Return the effective type (always a reference) of each element given
/// the type according to the type list.
RefType RefPackType::getElementRefTypeFor(Type elementType) {
  return RefType::get(elementType, getOrigin(), getAddressSpace());
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

static OptionalParseResult parseOptionalLITFuncType(AsmParser &p,
                                                    Type &signature) {
  llvm::SMLoc startLoc = p.getCurrentLocation();

  size_t numOriginDecls = 0;
  if (succeeded(p.parseOptionalLSquare()))
    if (p.parseInteger(numOriginDecls) || p.parseRSquare())
      return failure();

  TypedAttr captureOrigins;
  auto originSet = OriginSetType::get(p.getContext());
  if (succeeded(p.parseOptionalColon())) {
    if (parseParamValue(p, captureOrigins, originSet) || p.parseColon())
      return failure();
  } else {
    captureOrigins = OriginSetAttr::get({}, originSet);
  }
  bool isNestedOriginExclusivityCheckingDisabled =
      succeeded(p.parseOptionalKeyword("no_nested_origin_exclusivity"));

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaultPosArgs;
  SmallVector<TypedAttr> defaultKwOnlyArgs;
  SmallVector<ArgConvention> argConventions;
  SmallVector<VariadicKind> argVariadics;
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
    if (p.parseType(type) || parseConventionAndVariadicness(
                                 p, argConventions.emplace_back(),
                                 argVariadics.emplace_back(VariadicKind::None),
                                 origArgPackConvention, idx++))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, type,
                                         hasAddress(argConventions.back()))))
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
  OptionalParseResult result =
      parseOptionalSignatureValues(p, parseArg, functionType, effects,
                                   /*optionalResultList=*/false);
  if (!result.has_value())
    return std::nullopt;
  if (failed(*result))
    return failure();

  SmallVector<PassingKind> argPassingKinds;
  passingKindParser.populatePassingKinds(argPassingKinds);

  MLIRContext *ctx = p.getContext();
  auto metadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, argNames, argPassingKinds, defaultPosArgs,
                       defaultKwOnlyArgs, argVariadics, origArgPackConvention),
      numOriginDecls, captureOrigins,
      isNestedOriginExclusivityCheckingDisabled);
  signature =
      FuncType::getChecked([&] { return p.emitError(startLoc); }, functionType,
                           argConventions, effects, metadata);

  return success(!!signature);
}

static ParseResult parseLITFuncType(AsmParser &p, Type &signature) {
  OptionalParseResult result = parseOptionalLITFuncType(p, signature);
  if (result.has_value())
    return *result;
  return p.emitError(p.getCurrentLocation(), "expected LIT signature");
}

//===----------------------------------------------------------------------===//
// GeneratorType Parsing
//===----------------------------------------------------------------------===//

static ParseResult parseLITGenerator(AsmParser &p, Type &generator) {
  SmallVector<Type> inputParamTypes;
  PogListAttr paramListAttr = PogListAttr::get(p.getContext());
  Type body;
  if (LIT::parseOptionalParamSignature(p, inputParamTypes, paramListAttr))
    return failure();

  // Try to parse an unwrapped FnType fist.
  OptionalParseResult result = parseOptionalLITFuncType(p, body);
  if (result.has_value() && failed(*result))
    return failure();
  // If not a FnType, then parse as any other type.
  if (!result.has_value() && parseKGENType(p, body))
    return failure();

  generator = GeneratorType::get(inputParamTypes, body, paramListAttr);
  return success();
}

Type LITDialect::parseType(DialectAsmParser &p) const {
  llvm::SMLoc typeLoc = p.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  OptionalParseResult parseResult = generatedTypeParser(p, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Special alias for `!lit.fn` & `!lit.generator` types.
  if (mnemonic == "fn") {
    if (p.parseLess() || parseLITFuncType(p, genType) || p.parseGreater())
      return {};
    return genType;
  } else if (mnemonic == "generator") {
    if (p.parseLess() || parseLITGenerator(p, genType) || p.parseGreater())
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

//===----------------------------------------------------------------------===//
// LITGeneratorType
//===----------------------------------------------------------------------===//

LITGeneratorType::LITGeneratorType(GeneratorType gen) : GeneratorType(gen) {
  assert((!gen || ::isa_and_nonnull<PogListAttr>(gen.getMetadata())) &&
         "expected LIT generator metadata");
}

PogListAttr LITGeneratorType::getMetadata() {
  return ::cast<PogListAttr>(GeneratorType::getMetadata());
}

PogListAttr LITGeneratorType::getParamListAttrs() { return getMetadata(); }

StringAttr LITGeneratorType::getParamName(size_t idx) {
  return getMetadata().getName(idx);
}

//===----------------------------------------------------------------------===//
// FnType
//===----------------------------------------------------------------------===//

FnType::FnType(FuncType sig) : FuncType(sig) {
  assert((!sig || ::isa_and_nonnull<FnMetadataAttr>(sig.getMetadata())) &&
         "expected LIT function metadata");
}

FnMetadataAttr FnType::getMetadata() {
  return ::cast<FnMetadataAttr>(FuncType::getMetadata());
}

PogListAttr FnType::getArgListAttrs() {
  return getMetadata().getArgListAttrs();
}

StringAttr FnType::getArgName(size_t idx) {
  return getArgListAttrs().getName(idx);
}

TypedAttr FnType::getCaptureOrigins() {
  return getMetadata().getCaptureOrigins();
}

bool FnType::getIsNestedOriginExclusivityCheckingDisabled() {
  return getMetadata().getIsNestedOriginExclusivityCheckingDisabled();
}

ArrayRef<TypedAttr> FnType::getDefaultPosArgs() {
  return getMetadata().getDefaultPosArgs();
}

ArrayRef<TypedAttr> FnType::getDefaultKwOnlyArgs() {
  return getMetadata().getDefaultKwOnlyArgs();
}

/// Get the number of implicit origin decls this function type carries.
size_t FnType::getNumImplicitOriginDecls() {
  return getMetadata().getNumImplicitOriginDecls();
}

Type FnType::getUserResultType() {
  // If this function has a byref_result, return the reference element type.
  if (hasMemoryOnlyResult())
    return ::cast<RefType>(getArguments().back()).getElementType();
  return getResultType();
}

/// Substitute the specified implicit origin references into the specified
/// type, replacing them with `values` if they are at depth 0, or decrementing
/// their depth if not.  This returns the resultant FunctionType on success,
/// and invokes 'emitError'+returns null on error.
FunctionType FnType::substituteImplicitOriginsIntoValues(
    ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError) {
  assert(values.size() == getNumImplicitOriginDecls() &&
         "Incorrect # implicit origins specified");

  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = ::dyn_cast<ImplicitOriginRefAttr>(attr);
          ref && ref.getDepth() == depth) {
        if (ref.getIndex() >= values.size()) {
          emitError() << "implicit origin reference at depth " << depth
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

FnType FnType::getWithCaptureOrigins(TypedAttr origins) {
  return getWithMetadata(FnMetadataAttr::get(
      getArgListAttrs(), getNumImplicitOriginDecls(), origins,
      getIsNestedOriginExclusivityCheckingDisabled()));
}

bool FnType::isAnyVarArg(size_t index) {
  return getMetadata().isAnyVarArg(index);
}

bool FnType::isPosVarArg(size_t index) {
  return getMetadata().isPosVarArg(index);
}

/// For a PosVarArg, return the declared ArgConvention of the elements. For
/// example: fn x(inout *args: Int) is declared 'inout'.
ArgConvention FnType::getPosVarArgConvention(size_t index) {
  assert(isPosVarArg(index) && "isn't a positional vararg");
  return ::cast<VariadicType>(getArguments()[index]).getConvention();
}

bool FnType::isKwVarArg(size_t index) {
  return getMetadata().isKwVarArg(index);
}

bool FnType::isPack(size_t index) { return getMetadata().isPack(index); }

/// If the specified argument is a variadic pack, return the VariadicPack.
Type FnType::getIfVariadicPack(size_t index) {
  if (!isPack(index))
    return {};

  // Look through references to the VariadicPack type.
  auto type = getArguments()[index];
  if (hasAddress(getArgConvention(index)))
    type = ::cast<RefType>(type).getElementType();
  return type;
}

/// For a vararg, return the declared ArgConvention of the elements. For
/// example: fn x(mut *args: Int) is declared 'mut'.
ArgConvention FnType::getPackVarArgConvention(size_t index) {
  assert(getMetadata().isPack(index));
  return getArgListAttrs().getOrigPackConvention();
}

bool FnType::hasPackVarArgs() { return getMetadata().hasPackVarArgs(); }

bool FnType::hasKwVarArgs() { return getMetadata().hasKwVarArgs(); }

unsigned FnType::getErrorSlotOffset() {
  assert(isThrows() && "signature does not refer to a throwing function");
  return 1 + hasMemoryOnlyResult();
}

bool FnType::classof(FuncType type) {
  return ::isa_and_nonnull<FnMetadataAttr>(type.getMetadata());
}

bool FnType::classof(Type type) {
  if (auto sig = ::dyn_cast<FuncType>(type))
    return classof(sig);
  return false;
}

FnType FnType::get(MLIRContext *ctx, TypeRange inputs, TypeRange results,
                   size_t numImplicitOriginDecls) {
  auto funcType = FunctionType::get(ctx, inputs, results);

  size_t numInputs = funcType.getNumInputs();
  SmallVector<PogMetadataAttr> argPogs(
      numInputs,
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  auto metadata = FnMetadataAttr::get(PogListAttr::get(ctx, argPogs),
                                      numImplicitOriginDecls);
  return FuncType::get(funcType,
                       /*convs=*/{}, /*effects=*/{}, metadata);
}

//===----------------------------------------------------------------------===//
// FnTypeGeneratorType
//===----------------------------------------------------------------------===//

FnTypeGeneratorType::FnTypeGeneratorType(LITGeneratorType gen)
    : FuncTypeGeneratorType(gen) {
  assert((!gen || (::isa<FnType>(gen.getBody()))) &&
         "expected LIT generator wrapping LIT FnType");
}

FnTypeGeneratorType::FnTypeGeneratorType(FuncTypeGeneratorType gen)
    : FuncTypeGeneratorType(gen) {
  assert((!gen ||
          (::isa<LITGeneratorType>(gen) && ::isa<FnType>(gen.getBody()))) &&
         "expected LIT generator wrapping LIT FnType");
}

FnType FnTypeGeneratorType::getBody() {
  return ::cast<FnType>(GeneratorType::getBody());
}

PogListAttr FnTypeGeneratorType::getMetadata() {
  return ::cast<PogListAttr>(GeneratorType::getMetadata());
}

PogListAttr FnTypeGeneratorType::getParamListAttrs() { return getMetadata(); }

StringAttr FnTypeGeneratorType::getParamName(size_t idx) {
  return getMetadata().getName(idx);
}

FnMetadataAttr FnTypeGeneratorType::getFnMetadata() {
  return getBody().getMetadata();
}

PogListAttr FnTypeGeneratorType::getArgListAttrs() {
  return getBody().getArgListAttrs();
}

StringAttr FnTypeGeneratorType::getArgName(size_t idx) {
  return getArgListAttrs().getName(idx);
}

TypedAttr FnTypeGeneratorType::getCaptureOrigins() {
  return getBody().getCaptureOrigins();
}

bool FnTypeGeneratorType::getIsNestedOriginExclusivityCheckingDisabled() {
  return getBody().getIsNestedOriginExclusivityCheckingDisabled();
}

ArrayRef<TypedAttr> FnTypeGeneratorType::getDefaultPosArgs() {
  return getBody().getDefaultPosArgs();
}

ArrayRef<TypedAttr> FnTypeGeneratorType::getDefaultKwOnlyArgs() {
  return getBody().getDefaultKwOnlyArgs();
}

/// Get the number of implicit origin decls this function type carries.
size_t FnTypeGeneratorType::getNumImplicitOriginDecls() {
  return getBody().getNumImplicitOriginDecls();
}

Type FnTypeGeneratorType::getUserResultType() {
  return getBody().getUserResultType();
}

/// Substitute the specified implicit origin references into the specified
/// type, replacing them with `values` if they are at depth 0, or decrementing
/// their depth if not.  This returns the resultant FunctionType on success,
/// and invokes 'emitError'+returns null on error.
FunctionType FnTypeGeneratorType::substituteImplicitOriginsIntoValues(
    ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError) {
  return getBody().substituteImplicitOriginsIntoValues(values, emitError);
}

FnTypeGeneratorType
FnTypeGeneratorType::getWithCaptureOrigins(TypedAttr origins) {
  return getWithBody(getBody().getWithCaptureOrigins(origins));
}

bool FnTypeGeneratorType::isAnyVarArg(size_t index) {
  return getBody().isAnyVarArg(index);
}

bool FnTypeGeneratorType::isPosVarArg(size_t index) {
  return getBody().isPosVarArg(index);
}

/// For a PosVarArg, return the declared ArgConvention of the elements. For
/// example: fn x(inout *args: Int) is declared 'inout'.
ArgConvention FnTypeGeneratorType::getPosVarArgConvention(size_t index) {
  return getBody().getPosVarArgConvention(index);
}

bool FnTypeGeneratorType::isKwVarArg(size_t index) {
  return getBody().isKwVarArg(index);
}

bool FnTypeGeneratorType::isPack(size_t index) {
  return getBody().isPack(index);
}

/// If the specified argument is a variadic pack, return the VariadicPack.
Type FnTypeGeneratorType::getIfVariadicPack(size_t index) {
  return getBody().getIfVariadicPack(index);
}

/// For a PosVarArg, return the declared ArgConvention of the elements. For
/// example: fn x(inout *args: Int) is declared 'inout'.
ArgConvention FnTypeGeneratorType::getPackVarArgConvention(size_t index) {
  return getBody().getPackVarArgConvention(index);
}

bool FnTypeGeneratorType::hasPackVarArgs() {
  return getBody().hasPackVarArgs();
}

std::optional<size_t> FnTypeGeneratorType::findPackVarArgIndex() {
  size_t numUserArgs = getNumArguments() - hasMemoryOnlyResult();
  if (numUserArgs == 0)
    return std::nullopt;
  size_t lastUserArgIndex = numUserArgs - 1;
  if (isPack(lastUserArgIndex))
    return std::make_optional(lastUserArgIndex);
  return std::nullopt;
}

bool FnTypeGeneratorType::hasKwVarArgs() { return getBody().hasKwVarArgs(); }

unsigned FnTypeGeneratorType::getErrorSlotOffset() {
  assert(getBody().isThrows() &&
         "signature does not refer to a throwing function");
  return 1 + getBody().hasMemoryOnlyResult();
}

/// This method replaces direct uses of NAMED implicit origin declarations
/// with index-based references.  originDecls specifies the names of the
/// implicit origin decls to replace.
///
/// If indexOffset is subtracted from depth when set.
Type FnTypeGeneratorType::replaceImplicitOriginsWithIndexes(
    Type origType, ArrayRef<ParamDeclAttr> originDecls, size_t indexOffset) {

  // If there are no implicit origins, then this is a noop.
  if (originDecls.empty())
    return origType;

  // Replace named implicit origin parameter references with index-based
  // references in the signature.
  struct OriginDeclRemapper : IndexParameterReplacer<OriginDeclRemapper> {
    OriginDeclRemapper(size_t indexOffset) : indexOffset(indexOffset) {}

    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = ::dyn_cast<ParamDeclRefAttr>(attr)) {
        if (auto it = mapping.find(ref.getName()); it != mapping.end()) {
          // Subtract indexOffset because we may be replacing the signature
          // directly.
          size_t index = it->second;
          return ImplicitOriginRefAttr::get(depth - indexOffset, index,
                                            ref.getType());
        }
      }
      return nullptr;
    }

    size_t indexOffset;
    DenseMap<StringAttr, size_t> mapping;
  } remapper(indexOffset);
  for (auto [i, decl] : llvm::enumerate(originDecls))
    remapper.mapping.try_emplace(decl.getName(), i);
  return remapper.replace(origType);
}

/// This method replaces direct uses of NAMED implicit origin declarations
/// with index-based references corresponding to the signature. `originDecls`
/// specifies the names of the implicit origin decls.
FnTypeGeneratorType FnTypeGeneratorType::replaceImplicitOriginsWithIndexes(
    ArrayRef<ParamDeclAttr> originDecls) {
  assert(originDecls.size() == getBody().getNumImplicitOriginDecls() &&
         "Incorrect number of origin decls");
  return ::cast<FnTypeGeneratorType>(
      replaceImplicitOriginsWithIndexes(*this, originDecls, 1));
}

/// Reconstruct the signature using a list of named input parameters and
/// indices indicating which one of them are variadic. These parameters are
/// prepended to the current signature and references are remapped to index
/// references. An additional array of indices corresponding to variadic
/// parameters of the prepended parameters is also required.
FnTypeGeneratorType
FnTypeGeneratorType::prependParams(FnTypeGeneratorType sigGen,
                                   ArrayRef<ParamDeclAttr> parentParams,
                                   ArrayRef<VariadicKind> parentVariadics) {
  IndexRefRemapper remapper(parentParams, parentParams.size());
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : sigGen.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  FnType sig = sigGen.getBody();
  FnMetadataAttrInterface fnMetadata = remapper.replace(sig.getMetadata());
  GeneratorMetadataAttrInterface genMetadata =
      remapper.replace(sigGen.getMetadata().prependPosParams(
          parentParams.size(), parentVariadics));

  return FuncTypeGeneratorType::get(
      inputParamTypes, remapper.replace(sig.getValues()),
      sig.getArgConventions(), sig.getFnEffects(), fnMetadata, genMetadata);
}

bool FnTypeGeneratorType::classof(FuncTypeGeneratorType type) {
  return ::isa<LITGeneratorType>(type) && ::isa<FnType>(type.getBody());
}

bool FnTypeGeneratorType::classof(Type type) {
  if (auto sig = ::dyn_cast<FuncTypeGeneratorType>(type))
    return classof(sig);
  return false;
}

//===----------------------------------------------------------------------===//
// MetaTypeOf
//===----------------------------------------------------------------------===//

SymbolRefAttr StructMetaType::getSymbol() const {
  return getType().getValue().getValue();
}

TypeSignatureType StructMetaType::getSignature() const {
  return getType().getSignature();
}

ArrayRef<TypedAttr> StructMetaType::getParamValues() const {
  return getType().getParamValues();
}

StructMetaType StructMetaType::bindAll(ArrayRef<TypedAttr> values) const {
  return StructMetaType::get(getType().bindAll(values));
}

StructMetaType StructMetaType::bindUnbound(ArrayRef<TypedAttr> values) const {
  return StructMetaType::get(getType().bindUnbound(values));
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

Type LIT::getSignatureUserResultType(FnTypeGeneratorType sigType,
                                     ArrayRef<Type> argTypes, Type resultType) {
  // If this function has a byref_result, return the reference element type.
  if (sigType.getBody().hasMemoryOnlyResult())
    return cast<RefType>(argTypes.back()).getElementType();
  return resultType;
}

/// The Lit parser and KGEN have different semantics for binding function
/// argument and result types. The parser will evaluate 'apply' expressions, but
/// KGEN does not since it cannot always have access to a symbol table.
/// Specialize a signature type while rebinding the input parameter values to
/// the expected input parameter types.
std::pair<FnTypeGeneratorType, ParameterExprArrayAttr>
LIT::getUnboundSpecializedSignature(FnTypeGeneratorType type,
                                    ParameterExprArrayAttr bindings) {
  if (bindings.empty())
    return {type, bindings};

  // KGEN expects different bindings types than Lit can provide. Rebind the
  // parameters to the expected types.
  SmallVector<TypedAttr> unboundBindings;
  ParameterEvaluator evaluator;
  for (auto [binding, type] : llvm::zip(bindings, type.getInputParamTypes())) {
    TypedAttr value = binding;
    Type unboundType = evaluator.getReboundType(type);
    if (unboundType != value.getType())
      value = ParamOperatorAttr::get(POC::Rebind, value, unboundType);
    evaluator.addInputValue(value);
    unboundBindings.push_back(value);
  }
  type = type.getSpecializedGenerator(
      unboundBindings, [&]() -> InFlightDiagnostic {
        return mlir::emitError(UnknownLoc::get(type.getContext()));
      });
  assert(type && "bad bindings specified");
  return {type,
          ParameterExprArrayAttr::get(type.getContext(), unboundBindings)};
}
