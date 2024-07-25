//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/STLExtras.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <llvm/ADT/STLExtras.h>
#include <numeric>

using namespace M;
using namespace KGEN;

// Provide implementations for the enums we use.
#include "KGEN/KGENDialect/KGENEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace mlir {
/// Parse a dtype.
template <>
struct FieldParser<KGENDType> {
  static FailureOr<KGENDType> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    return KGENDType::getFromString(value);
  }
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// KGENDialect attribute support
//===----------------------------------------------------------------------===//

void KGENDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// VariadicAttr
//===----------------------------------------------------------------------===//

/// The variadic attribute is a constant if all element values are constants.
bool VariadicAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant) &&
         !isParameterizedType(getType());
}

LogicalResult VariadicAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   ArrayRef<TypedAttr> values,
                                   VariadicType type) {
  Type elementType = type.getElementType();
  for (auto [idx, value] : llvm::enumerate(values))
    if (value.getType() != elementType)
      return emitError() << "variadic sequence element #" << idx << " has type "
                         << value.getType() << " but expected " << elementType;
  return success();
}

static ParseResult parseVariadicValue(AsmParser &p,
                                      SmallVector<TypedAttr> &values,
                                      VariadicType type) {
  return p.parseCommaSeparatedList([&] {
    return parseParamValue(p, values.emplace_back(), type.getElementType());
  });
}

OptionalParseResult VariadicType::parseValue(AsmParser &p,
                                             TypedAttr &value) const {
  if (failed(p.parseOptionalLSquare()))
    return std::nullopt;
  if (succeeded(p.parseOptionalRSquare())) {
    value = VariadicAttr::get({}, *this);
    return mlir::success();
  }
  SmallVector<TypedAttr> values;
  if (failed(parseVariadicValue(p, values, *this)))
    return failure();
  value = VariadicAttr::get(values, *this);
  return p.parseRSquare();
}

LogicalResult VariadicType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto variadic = llvm::dyn_cast<VariadicAttr>(value);
  if (!variadic)
    return failure();
  p << '[';
  llvm::interleaveComma(variadic.getValues(), p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ']';
  return success();
}

//===----------------------------------------------------------------------===//
// UnknownAttr
//===----------------------------------------------------------------------===//

bool UnknownAttr::isConstant() const { return !isParameterizedType(getType()); }

//===----------------------------------------------------------------------===//
// UnboundAttr
//===----------------------------------------------------------------------===//

bool UnboundAttr::isConstant() const { return !isParameterizedType(getType()); }

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

Type NoneAttr::getType() const { return KGEN::NoneType::get(getContext()); }

bool NoneAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// A parameter reference forms the basis of a non-constant parameter attribute.
bool ParamDeclRefAttr::isConstant() const { return false; }

/// Sort the parameter references by name.
std::optional<bool> ParamDeclRefAttr::isLessThan(Attribute rhs) const {
  if (auto ref = llvm::dyn_cast<ParamDeclRefAttr>(rhs))
    return getName().getValue() < ref.getName().getValue();
  // Otherwise, named parameters are always to the right.
  return false;
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttr
//===----------------------------------------------------------------------===//

/// A parameter reference is not a constant by definition.
bool ParamIndexRefAttr::isConstant() const { return false; }

/// Sort index references by index then kind.
std::optional<bool> ParamIndexRefAttr::isLessThan(Attribute rhs) const {
  auto ref = ::dyn_cast<ParamIndexRefAttr>(rhs);
  if (!ref)
    return false;
  return std::make_tuple(getDepth(), getIndex(), getIsResult()) <
         std::make_tuple(ref.getDepth(), ref.getIndex(), ref.getIsResult());
}

//===----------------------------------------------------------------------===//
// TypeConstantAttr
//===----------------------------------------------------------------------===//

Attribute TypeConstantAttr::parse(AsmParser &p, Type type) {
  if (p.parseLess())
    return {};

  TypedAttr value;
  OptionalParseResult result =
      parseTypeValueBody(p, value, type, parseOptionalKGENType);
  if (!result.has_value()) {
    p.emitError(p.getCurrentLocation(), "expected a type value");
    return {};
  }

  if (failed(*result) || p.parseGreater())
    return {};
  return value;
}

void TypeConstantAttr::print(AsmPrinter &p) const {
  p << '<';
  void (*typePrinter)(AsmPrinter &, Type) = &printKGENType; // Select overload.
  printTypeValueBody(p, *this, typePrinter);
  p << '>';
}

TypedAttr TypeConstantAttr::get(MLIRContext *ctx, Type typeValue, Type mlirType,
                                Type type, VTableAttr vtable) {
  // If this is a trivial mlir Type (i.e. has identical type & value
  // representation), and the trivial type is a ParamRefType, then we're
  // unwrapping a wrapper. Remove this to keep the types canonical.
  if (mlirType == typeValue && vtable.getEntries().empty())
    if (auto refType = ::dyn_cast<ParamRefType>(mlirType))
      return refType.getParam();

  return Base::get(ctx, typeValue, mlirType, type, vtable);
}

TypedAttr TypeConstantAttr::get(Type typeValue, Type mlirType, Type type,
                                VTableAttr vtable) {
  return get(mlirType.getContext(), typeValue, mlirType, type, vtable);
}

TypedAttr TypeConstantAttr::get(Type typeValue, Type mlirType, Type type) {
  return get(typeValue, mlirType, type, VTableAttr::get(type.getContext(), {}));
}

TypedAttr TypeConstantAttr::get(Type mlirType, Type type, VTableAttr vtable) {
  return get(mlirType, mlirType, type, vtable);
}

TypedAttr TypeConstantAttr::get(Type mlirType, Type type) {
  return get(mlirType, mlirType, type);
}

TypeConstantAttr TypeConstantAttr::getFromBytecode(Type typeValue,
                                                   Type mlirType, Type type,
                                                   VTableAttr vtable) {
  return Base::get(mlirType.getContext(), typeValue, mlirType, type, vtable);
}

bool TypeConstantAttr::isConstant() const {
  return !isParameterizedType(getMlirType());
}

bool TypeConstantAttr::hasIdenticalRepresentation() {
  return getMlirType() == getTypeValue() && getVTable().getEntries().empty();
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

Type DTypeConstantAttr::getType() const { return DTypeType::get(getContext()); }

bool DTypeConstantAttr::isConvertibleTo(Type type) {
  KGENDType dtype = getDType();

  // Bool can only be `i1`.
  if (dtype.isBool())
    return type.isSignlessInteger(1);

  // Index DType can only be the mlir `index` type.
  if (dtype.isIndex())
    return type.isIndex();

  // Integer dtypes can be converted to MLIR integers of the same width and
  // un-opposing signedness; signed integer dtypes can be converted to signless
  // and signed MLIR integer types but not unsigned.
  if (dtype.isInt()) {
    auto intType = llvm::dyn_cast<IntegerType>(type);
    if (!intType || intType.getWidth() != dtype.getWidthInBits())
      return false;
    return intType.isSignless() || intType.isSigned() == dtype.isSInt();
  }

  // Floating point dtypes can be converted to equivalent MLIR float types.
  if (dtype.isFloat()) {
    if (auto fpType = llvm::dyn_cast<FloatType>(type))
      return areEquivalentFloatTypes(dtype, fpType);
    return false;
  }

  return false;
}

bool DTypeConstantAttr::isConvertibleFrom(Type type) {
  KGENDType dtype = getDType();

  if (dtype.isBool())
    return llvm::isa<IntegerType>(type);

  // Signless integers cannot be converted.
  if (type.isSignlessInteger() && !dtype.isIndex())
    return false;

  // Index dtypes can be converted if the type is an IndexType.
  if (dtype.isIndex() && llvm::isa<IndexType>(type))
    return true;

  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    if (dtype.isIndex())
      return true;
    // Integers can be converted to dtypes of the same width and signedness.
    if (dtype.isInt() && dtype.getWidthInBits() == intType.getWidth() &&
        dtype.isSInt() == intType.isSigned())
      return true;
    // Otherwise, we risk loosing bits, so we conservatively disallow.
    return false;
  }

  // Floating point types can be converted to equivalent dtypes.
  if (auto fpType = llvm::dyn_cast<FloatType>(type))
    return dtype.isFloat() && areEquivalentFloatTypes(dtype, fpType);

  return false;
}

/// Always a constant by definition.
bool DTypeConstantAttr::isConstant() const { return true; }

/// Sort by dtype value.
std::optional<bool> DTypeConstantAttr::isLessThan(Attribute rhs) const {
  if (auto dtype = llvm::dyn_cast<DTypeConstantAttr>(rhs))
    return getDType().getValue() < dtype.getDType().getValue();
  return true;
}

//===----------------------------------------------------------------------===//
// IntLiteralAttr
//===----------------------------------------------------------------------===//

static ParseResult parseIntLiteral(AsmParser &p, IPInt &result) {
  APInt resultAP;
  OptionalParseResult parseResult = p.parseInteger(resultAP);
  if (!parseResult.has_value() || failed(*parseResult)) {
    result = {};
    return failure();
  }
  result = IPInt(resultAP);
  return success();
}

static void printIntLiteral(AsmPrinter &p, const IPInt &value) {
  p.getStream() << value;
}

Type IntLiteralAttr::getType() const {
  return IntLiteralType::get(this->getContext());
}

bool IntLiteralAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// FloatLiteralAttr
//===----------------------------------------------------------------------===//

Attribute FloatLiteralAttr::parse(AsmParser &p, Type type) {
  if (p.parseLess())
    p.emitError(p.getCurrentLocation(), "expected '<' character");
  std::optional<IPRational> rational;
  FloatLiteralSpecialValuesAttr specialAttr;

  // Try to parse rational number first, then fall back to parsing special
  // value.
  APInt numerator;
  OptionalParseResult isRational = p.parseOptionalInteger(numerator);
  if (isRational.has_value() && !isRational.value()) {
    // MLIR's AsmParser doesn't have `parseSlash` or a more generic way to parse
    // literal strings/characters, so we will use the pipe "|" character
    // instead. https://github.com/modularml/modular/issues/23387
    APInt denominator;
    if (p.parseVerticalBar() || p.parseInteger(denominator)) {
      p.emitError(p.getCurrentLocation(),
                  "expected rational number with pipe in parens");
      return {};
    }
    if (denominator == 0) {
      p.emitError(p.getCurrentLocation(),
                  "expected rational number with non-zero denominator");
    }
    rational = IPRational(numerator, denominator);
    specialAttr = FloatLiteralSpecialValuesAttr::get(
        p.getContext(), FloatLiteralSpecialValues::Normal);
  } else if (p.parseCustomAttributeWithFallback(specialAttr)) {
    p.emitError(p.getCurrentLocation(),
                "expected FloatLiteralSpecialValueAttr");
    return {};
  }

  if (p.parseGreater()) {
    p.emitError(p.getCurrentLocation(), "expected '>' character");
  }

  return FloatLiteralAttr::get(p.getContext(), specialAttr, rational);
}

void FloatLiteralAttr::print(AsmPrinter &p) const {
  p.getStream() << "<";
  if (getSpecial().getValue() == FloatLiteralSpecialValues::Normal) {
    assert(getRational().has_value() &&
           "rational has value when special value is normal");
    p.getStream() << getRational().value();
  } else {
    p.getStream() << getSpecial().getValue();
  }
  p.getStream() << ">";
}

Type FloatLiteralAttr::getType() const {
  return FloatLiteralType::get(this->getContext());
}

bool FloatLiteralAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// VTableAttr
//===----------------------------------------------------------------------===//

static ParseResult parseVTableEntry(AsmParser &p, StringAttr &name,
                                    TypedAttr &method) {
  std::string nameStr;
  if (p.parseString(&nameStr))
    return failure();
  name = StringAttr::get(p.getContext(), nameStr);
  Type signature;
  if (p.parseColon() || parseSignature(p, signature) || p.parseEqual() ||
      parseParamValue(p, method, signature))
    return failure();
  return success();
}

static void printVTableEntry(AsmPrinter &p, StringAttr name, TypedAttr method) {
  p.printString(name.getValue());
  p << " : ";
  printSignature(p, method.getType());
  p << " = ";
  printParamValue(p, method);
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr
//===----------------------------------------------------------------------===//

/// This symbol is a constant its bindings are constants.
bool SymbolConstantAttr::isConstant() const {
  return llvm::all_of(getParamValues(), ParameterAttr::isSimpleConstant) &&
         !isParameterizedType(getType());
}

LogicalResult
SymbolConstantAttr::verifySymbolUses(Operation *module,
                                     mlir::LockedSymbolTableCollection &symtab,
                                     Location loc) const {
  VerboseCompilerTimeTraceScope traceScope(
      "SymbolConstantAttr::verifySymbolUses");

  // Build the signature of the referenced symbol.
  SymbolRefAttr symbol = getSymbol();
  SmallVector<Operation *> symbolOps;
  {
    VerboseCompilerTimeTraceScope traceScope("lookupSymbolIn");
    if (failed(symtab.lookupSymbolIn(module, symbol, symbolOps)))
      return emitError(loc)
             << symbol << " does not reference a KGEN declaration";
  }

  // The leaf symbol must refer to a function.
  auto func = ::dyn_cast<FuncInterface>(symbolOps.back());
  if (!func)
    return emitError(loc) << symbol << " does not reference a KGEN function";

  // Everything else must be a declaration.
  for (Operation *op : llvm::drop_end(symbolOps)) {
    if (!::isa<DeclInterface>(op)) {
      return emitError(loc)
             << "symbol @" << ::cast<mlir::SymbolOpInterface>(op).getName()
             << " does not reference a KGEN declaration";
    }
  }

  SignatureType declSignature;
  if (symbolOps.size() == 1) {
    declSignature = func.getSignature().getSpecializedSignature(
        getParamValues(), [&] { return emitError(loc); });
  } else {
    // Collect the contextual parameter values.
    SmallVector<ParamDeclAttr> paramDecls;
    for (Operation *op : llvm::drop_end(symbolOps))
      llvm::append_range(paramDecls,
                         ::cast<DeclInterface>(op).getInputParams());

    IndexRefRemapper remapper(paramDecls, /*resultParams=*/{},
                              paramDecls.size());
    SignatureType baseSig = func.getSignature();
    SmallVector<Type> inputParamTypes;
    for (ParamDeclAttr param : paramDecls)
      inputParamTypes.push_back(remapper.replace(param.getType()));
    for (Type type : baseSig.getInputParamTypes())
      inputParamTypes.push_back(remapper.replace(type));

    FnMetadataAttrInterface metadata = baseSig.getMetadata();
    if (metadata) {
      metadata = remapper.replace(
          metadata.prependPosParamsFromOps(ArrayRef(symbolOps).drop_back()));
    }

    declSignature = SignatureType::getSpecializedSignature(
        getParamValues(), [&] { return emitError(loc); }, inputParamTypes,
        remapper.replace(baseSig.getResultParamTypes()),
        remapper.replace(baseSig.getValues()), baseSig.getArgConventions(),
        baseSig.getFnEffects(), metadata);
  }
  if (!declSignature)
    return failure();

  // Parameter types match exactly.  We could support higher order rebinding
  // if there is a need.
  return verifyDeclSignaturesMatch("symbol use", getType(), loc,
                                   symbol.getLeafReference(), declSignature,
                                   func->getLoc());
}

ParseResult parseColonTypeSymbolConstant(AsmParser &p,
                                         SymbolConstantAttr &value) {
  mlir::SMLoc loc = p.getCurrentLocation();

  TypedAttr typedAttr;
  if (parseColonTypeParamValue(p, typedAttr))
    return failure();

  if (auto symbol = mlir::dyn_cast<SymbolConstantAttr>(typedAttr)) {
    value = symbol;
    return success();
  }

  return p.emitError(loc) << "symbol constant expected, got" << typedAttr;
}

void printColonTypeSymbolConstant(AsmPrinter &p, SymbolConstantAttr value) {
  printColonTypeParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// TargetParamAttr
//===----------------------------------------------------------------------===//

Attribute TargetParamAttr::parse(AsmParser &p, Type type) {
  auto targetType = llvm::dyn_cast_or_null<TargetType>(type);
  if (!targetType) {
    p.emitError(p.getCurrentLocation(),
                "target parameter expected a target type");
    return {};
  }

  // Otherwise, parse the whole target info attribute.
  TargetInfoAttr target;
  if (p.parseCustomAttributeWithFallback(target))
    return {};
  return TargetParamAttr::get(target);
}

void TargetParamAttr::print(AsmPrinter &p) const { getTarget().print(p); }

Type TargetParamAttr::getType() const { return TargetType::get(getContext()); }

/// Always a constant.
bool TargetParamAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// StructDefParamRefAttr
//===----------------------------------------------------------------------===//

/// A parameter reference forms the basis of a non-constant parameter attribute.
bool StructDefParamRefAttr::isConstant() const { return false; }

/// Sort the parameter references by name.
std::optional<bool> StructDefParamRefAttr::isLessThan(Attribute rhs) const {
  if (auto ref = llvm::dyn_cast<StructDefParamRefAttr>(rhs))
    return getName().getValue() < ref.getName().getValue();
  // Otherwise, named parameters are always to the right.
  return false;
}

//===----------------------------------------------------------------------===//
// StructDefAttr
//===----------------------------------------------------------------------===//

static ParseResult parseParamNamesAndTypes(AsmParser &p,
                                           SmallVector<StringAttr> &names,
                                           SmallVector<Type> &types) {
  // Empty list.
  if (failed(p.parseOptionalLSquare()))
    return success();

  if (p.parseCommaSeparatedList([&]() {
        StringAttr name;
        Type type;
        if (parseParamName(p, name) || parseColonTypeOrIndex(p, type))
          return failure();
        names.push_back(name);
        types.push_back(type);
        return mlir::success();
      }) ||
      p.parseRSquare())
    return failure();
  return success();
}

static void printParamNamesAndTypes(AsmPrinter &p, ArrayRef<StringAttr> names,
                                    ArrayRef<Type> types) {
  if (names.empty())
    return;

  p << '[';
  llvm::interleaveComma(llvm::zip(names, types), p,
                        [&p](const std::tuple<StringAttr, Type> &decl) {
                          printParamName(p, std::get<0>(decl));
                          printColonTypeOrIndex(p, std::get<1>(decl));
                        });
  p << ']';
}

static ParseResult
parseStructDefFields(AsmParser &p, SmallVector<StructDefFieldAttr> &fields) {
  MLIRContext *ctx = p.getContext();
  return p.parseCommaSeparatedList([&]() {
    StringAttr name;
    Type type;
    if (parseParamName(p, name) || p.parseColon() || parseKGENType(p, type))
      return failure();
    fields.push_back(StructDefFieldAttr::get(ctx, name, type));
    return mlir::success();
  });
}

static void printStructDefFields(AsmPrinter &p,
                                 ArrayRef<StructDefFieldAttr> fields) {
  llvm::interleaveComma(fields, p, [&](StructDefFieldAttr field) {
    printParamName(p, field.getName());
    p << ": ";
    printKGENType(p, field.getType());
  });
}

StructDefAttr StructDefAttr::get(StringAttr name,
                                 ArrayRef<StringAttr> inputParamNames,
                                 ArrayRef<Type> inputParamTypes,
                                 ArrayRef<StructDefFieldAttr> fields,
                                 bool isMemoryOnly) {
  return get(name.getContext(), name, inputParamNames, inputParamTypes, fields,
             isMemoryOnly);
}

LogicalResult StructDefAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    ArrayRef<StringAttr> inputParamNames, ArrayRef<Type> inputParamTypes,
    ::llvm::ArrayRef<StructDefFieldAttr> fields, bool isMemoryOnly) {
  if (inputParamNames.size() != inputParamTypes.size()) {
    return emitError() << "#kgen.struct_def parameter name and parameter "
                          "type length mismatch. Expected "
                       << inputParamNames.size() << ", got "
                       << inputParamTypes.size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

static ParseResult parseStructElements(AsmParser &p,
                                       SmallVector<TypedAttr> &values,
                                       StructType type) {
  return failableInterleave(
      type.getElementTypes(),
      [&](Type type) {
        return parseParamValue(p, values.emplace_back(), type);
      },
      [&] { return p.parseComma(); });
}

static void printStructElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                                StructType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

OptionalParseResult StructType::parseValue(AsmParser &p,
                                           TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return std::nullopt;
  SmallVector<TypedAttr> values;
  if (failed(parseStructElements(p, values, *this)))
    return failure();
  value = StructAttr::get(values, *this);
  return p.parseRBrace();
}

LogicalResult StructType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto structAttr = ::dyn_cast<StructAttr>(value);
  if (!structAttr)
    return failure();
  p << "{ ";
  llvm::interleaveComma(structAttr.getValues(), p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << " }";
  return mlir::success();
}

/// The struct attribute is a constant if all element values are constants.
bool StructAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

LogicalResult StructAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<TypedAttr> values, StructType type) {
  ArrayRef<Type> types = type.getElementTypes();
  if (types.size() != values.size())
    return emitError() << "struct attribute type requires " << types.size()
                       << " elements but value has " << values.size();
  for (auto [idx, value, type] :
       llvm::zip(llvm::seq<unsigned>(0, types.size()), values, types)) {
    if (value.getType() != type) {
      return emitError() << "struct element #" << idx << " has type "
                         << value.getType() << " but expected " << type;
    }
  }
  return success();
}

StructAttr StructAttr::get(ArrayRef<TypedAttr> values) {
  assert(!values.empty() && "expected at least one value");
  SmallVector<Type> types;
  types.reserve(values.size());
  for (TypedAttr value : values)
    types.push_back(value.getType());
  return StructAttr::get(values, StructType::get(types));
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

TypedAttr StructExtractAttr::get(TypedAttr structValue, unsigned fieldNo) {
  auto structType = ::cast<StructType>(structValue.getType());
  assert(fieldNo < structType.getElementTypes().size() &&
         "struct extract index out of range");
  return get(structValue.getContext(), structValue, fieldNo,
             structType.getElementTypes()[fieldNo]);
}

TypedAttr StructExtractAttr::get(MLIRContext *context, TypedAttr structValue,
                                 unsigned fieldNo, Type resultType) {
  if (auto value = dyn_cast_if_present<StructAttr>(structValue))
    return value.getValues()[fieldNo];
  if (::isa<UnknownAttr>(structValue))
    return UnknownAttr::get(resultType);

  return Base::get(context, structValue, fieldNo, resultType);
}

StructExtractAttr StructExtractAttr::getFromBytecode(TypedAttr structValue,
                                                     unsigned fieldNo,
                                                     Type resultType) {
  return Base::get(resultType.getContext(), structValue, fieldNo, resultType);
}

//===----------------------------------------------------------------------===//
// PackAttr
//===----------------------------------------------------------------------===//

static ParseResult
parsePackElements(AsmParser &p, SmallVector<TypedAttr> &values, PackType type) {
  auto variadic = type.getVariadicIfResolved();
  if (!variadic)
    return p.emitError(p.getCurrentLocation())
           << "pack attribute expected a variadic constant type, but got "
           << type.getVariadic();

  return failableInterleave(
      variadic.getValues(),
      [&](TypedAttr eltType) {
        return parseParamValue(p, values.emplace_back(),
                               ParamRefType::get(eltType));
      },
      [&] { return p.parseComma(); });
}

static void printPackElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                              PackType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

OptionalParseResult PackType::parseValue(AsmParser &p, TypedAttr &value) const {
  if (failed(p.parseOptionalLess()))
    return std::nullopt;
  SmallVector<TypedAttr> values;
  if (failed(parsePackElements(p, values, *this)))
    return failure();

  value = PackAttr::get(values, *this);
  return p.parseGreater();
}

LogicalResult PackType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto packAttr = ::dyn_cast<PackAttr>(value);
  if (!packAttr)
    return failure();

  p << "<";
  printPackElements(p, packAttr.getValues(), *this);
  p << ">";
  return success();
}

LogicalResult PackAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<TypedAttr> values, PackType packType) {
  auto variadic = packType.getVariadicIfResolved();
  if (!variadic)
    return emitError()
           << "pack attribute expected a variadic constant type, but got "
           << packType.getVariadic();

  ArrayRef<TypedAttr> expected = variadic.getValues();
  if (values.size() != expected.size())
    return emitError() << "pack attribute type requires " << expected.size()
                       << " elements, but got " << values.size();
  // Verify the constant elements have the right type.
  for (auto [i, value, typeAttr] :
       llvm::zip(llvm::seq<size_t>(0, expected.size()), values, expected))
    if (value.getType() != ParamRefType::get(typeAttr))
      return emitError() << "pack attribute element #" << i << " has type "
                         << value.getType() << " but expected " << typeAttr;
  return success();
}

bool PackAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

//===----------------------------------------------------------------------===//
// VariantAttr
//===----------------------------------------------------------------------===//

LogicalResult VariantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr value, unsigned index,
                                  VariantType type) {
  if (index >= type.getNumTypes())
    return emitError() << "variant index " << index << " is out of bounds";
  if (type.getType(index) == value.getType())
    return success();
  return emitError() << "variant attribute value type " << value.getType()
                     << " does not match type at index " << index
                     << " which is " << type.getType(index);
}

/// The variant attribute is a constant if the value type is a constant and its
/// type is not parameterized. It is possible to materialize a constant value
/// for a parametric variant type.
bool VariantAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getValue()) &&
         !isParameterizedType(getType());
}

//===----------------------------------------------------------------------===//
// EnvAttr
//===----------------------------------------------------------------------===//

LogicalResult EnvAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DictionaryAttr values) {
  // Only index, string, and unit attributes are allowed.
  for (const NamedAttribute &attr : values) {
    Attribute value = attr.getValue();
    if (auto intVal = ::dyn_cast<IntegerAttr>(value)) {
      if (!::isa<IndexType>(intVal.getType()))
        return emitError() << "environment value " << attr.getName()
                           << " is an integer not of `index` type";
    } else if (auto strVal = ::dyn_cast<StringAttr>(value)) {
      if (!::isa<StringType>(strVal.getType()))
        return emitError() << "environment value " << attr.getName()
                           << " is a string not of `!kgen.string` type";
    } else if (!::isa<mlir::UnitAttr>(value)) {
      return emitError() << "environment value " << attr.getName()
                         << " is neither an index, string, or unit attribute";
    }
  }
  return success();
}

ErrorOr<EnvAttr> EnvAttr::parseDefines(MLIRContext *ctx,
                                       ArrayRef<std::string> defines) {
  NamedAttrList attrs;
  Builder b(ctx);
  for (StringRef define : defines) {
    size_t idx = define.find("=");
    // If '=' is not present in the string, then this is a unit define.
    if (idx == StringRef::npos) {
      if (attrs.set(define, b.getUnitAttr()))
        return Error("'" + define + "' was defined more than once");
      continue;
    }
    StringRef name = define.slice(0, idx);
    StringRef value = define.slice(idx + 1, StringRef::npos);

    // Try to convert the value to an integer.
    APInt intVal(IndexType::kInternalStorageBitWidth, 0);
    if (!value.getAsInteger(/*Radix=*/10, intVal)) {
      if (attrs.set(name, b.getIndexAttr(intVal.getSExtValue())))
        return Error("'" + define + "' was defined more than once");
      continue;
    }

    // Otherwise, use it as a string value.
    if (attrs.set(name, StringAttr::get(value, StringType::get(ctx))))
      return Error("'" + define + "' was defined more than once");
  }

  return EnvAttr::get(attrs.getDictionary(ctx));
}

EnvAttr EnvAttr::extend(EnvAttr attr) {
  NamedAttrList values = getValues();
  for (NamedAttribute val : attr.getValues()) {
    // If the key exists, then we remove it so that we overwrite.
    if (values.getNamed(val.getName()))
      values.erase(val.getName());

    values.append(val);
  }

  return EnvAttr::get(values.getDictionary(attr.getContext()));
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

ParamOperatorAttr
ParamOperatorAttr::getFromBytecode(POC opcode, ArrayRef<TypedAttr> operands,
                                   Type type) {
  return Base::get(type.getContext(), opcode, operands, type);
}

static FailureOr<SignatureType>
verifyBindSignature(ArrayRef<TypedAttr> operands,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (operands.empty())
    return emitError() << "'bind_signature' requires a function parameter";
  auto signature = dyn_cast<SignatureType>(operands[0].getType());
  if (!signature)
    return emitError()
           << "first operand of 'bind_signature' must have signature type";

  // Get the specialized version of the signature with all the known parameters
  // substituted in.
  auto result =
      signature.getSpecializedSignature(operands.drop_front(), emitError);
  if (!result)
    return failure();

  return result;
}

/// The 'apply' operator is the only way to call a signature value inside a
/// parameter expression. Therefore, it is the only place where an index
/// parameter reference can cross upwards across a signature. We need to
/// decrement any index references in the result type of the signature because
/// we are pulling it out of the signature.
static Type upbindApplyResult(Type resultType) {
  IndexDepthAdjuster adjuster(/*adjustDepth=*/-1);
  return adjuster.replace(resultType);
}

/// Verify the apply and apply_result ParameterOperatorExprs.
static LogicalResult
verifyApplyLike(ArrayRef<TypedAttr> operands, bool isApplyResult,
                function_ref<InFlightDiagnostic()> emitError) {
  StringRef prefix = isApplyResult ? "'apply_result_slot' " : "'apply' ";
  if (operands.empty())
    return emitError() << prefix << "expected a function parameter";

  auto sig = cast<SignatureType>(operands.front().getType());
  if (!sig.getResultParamTypes().empty() || !sig.getInputParamTypes().empty())
    return emitError() << prefix << "function cannot be parametric";

  // Verify the inputs.
  // Drop the callee and the result slot type for apply_result.
  operands = operands.drop_front();
  ArrayRef<Type> inputTypes = sig.getArguments();
  if (isApplyResult) {
    if (sig.hasInitSelfArg())
      inputTypes = inputTypes.drop_front();
    else
      inputTypes = inputTypes.drop_back();
  }

  if (operands.size() != inputTypes.size()) {
    return emitError() << "'apply' function expected " << inputTypes.size()
                       << " inputs but got " << operands.size() << "\n";
  }
  for (auto [i, operand, type] : llvm::enumerate(operands, inputTypes)) {
    Type expected = upbindApplyResult(type);
    if (operand.getType() != expected) {
      return emitError() << "'apply' operand #" << i << " type "
                         << operand.getType()
                         << " does not match expected type " << expected;
    }
  }

  return success();
}

static LogicalResult verifyApply(ArrayRef<TypedAttr> operands, Type type,
                                 function_ref<InFlightDiagnostic()> emitError) {
  if (failed(verifyApplyLike(operands, /*isApplyResult=*/false, emitError)))
    return failure();

  // Verify the result.
  auto sig = cast<SignatureType>(operands.front().getType());
  if (sig.getResults().size() != 1)
    return emitError() << "'apply' function must return one result";
  Type resultType = upbindApplyResult(sig.getResults().front());
  if (type != resultType)
    return emitError() << "'apply' function result type must be " << type
                       << " but got " << resultType;

  return success();
}

static LogicalResult
verifyApplyResultSlot(ArrayRef<TypedAttr> operands, Type type,
                      function_ref<InFlightDiagnostic()> emitError) {
  if (failed(verifyApplyLike(operands, /*isApplyResult=*/true, emitError)))
    return failure();

  auto sig = cast<SignatureType>(operands.front().getType());
  // TODO: Cannot check !lit.ref reference types in KGEN.
  auto resultArgType = sig.hasInitSelfArg() ? sig.getArguments().front()
                                            : sig.getArguments().back();
  if (auto resultPtr = dyn_cast<PointerType>(resultArgType)) {
    auto expectedResult = resultPtr.getElementType();
    if (expectedResult != type)
      return emitError() << "'apply_result' function result type must be "
                         << expectedResult << " but got " << type;
  }
  return success();
}

static LogicalResult
verifyGetTypeMethod(ArrayRef<TypedAttr> operands, Type type,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (operands.size() != 2)
    return emitError() << "'get_type_method' requires 2 operands";
  if (!isa<StringType>(operands[1].getType()))
    return emitError() << "'get_type_method' second operand should be a string";
  if (!isa<SignatureType>(type))
    return emitError() << "'get_type_method' result should be a type signature";
  return success();
}

static LogicalResult
verifyVariadicPtrMap(ArrayRef<TypedAttr> operands, Type type,
                     function_ref<InFlightDiagnostic()> emitError) {
  if (operands.size() != 2)
    return emitError() << "'variadic_ptr_map' requires 2 operands";

  auto srcVariadic = dyn_cast<VariadicType>(operands[0].getType());
  if (!srcVariadic ||
      !isa<TypeType, ParamRefType>(srcVariadic.getElementType()) ||
      type != srcVariadic)
    return emitError() << "'variadic_ptr_map' operand should have "
                          "!kgen.variadic<!kgen.type> type, not "
                       << operands[0].getType();
  if (!operands[1].getType().isIndex())
    return emitError()
           << "'variadic_ptr_map' addr space operand should have 'index' type";

  return success();
}

static LogicalResult
verifyVariadicPtrRemoveMap(ArrayRef<TypedAttr> operands, Type type,
                           function_ref<InFlightDiagnostic()> emitError) {
  if (operands.size() != 1)
    return emitError() << "'variadic_ptrremove_map' requires 1 operand";

  auto srcVariadic = dyn_cast<VariadicType>(operands[0].getType());
  if (!srcVariadic || // May still be parametric
      !isa<TypeType>(srcVariadic.getElementType()))
    return emitError() << "'variadic_ptrremove_map' operand should have "
                          "!kgen.variadic<!kgen.type> type, not "
                       << operands[0].getType();
  auto dstVariadic = dyn_cast<VariadicType>(type);
  if (!dstVariadic || !isa<TypeType>(dstVariadic.getElementType()))
    return emitError() << "'variadic_ptrremove_map' result should be "
                          "!kgen.variadic<!kgen.type> type, not "
                       << type;
  return success();
}

LogicalResult ParamOperatorAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, POC opcode,
    ArrayRef<TypedAttr> operands, Type type) {
  // All the operand types must match except for these operators.
  switch (opcode) {
  case POC::Cond:
  case POC::TargetHasFeature:
  case POC::TargetGetField:
  case POC::GetEnv:
  case POC::GetSizeOf:
  case POC::GetAlignOf:
  case POC::BindSignature:
  case POC::Apply:
  case POC::ApplyResultSlot:
  case POC::Rebind:
  case POC::VariadicGet:
  case POC::CompileAssembly:
  case POC::GetLinkageName:
  case POC::GetTypeMethod:
  case POC::PtrBitcast:
  case POC::VariadicPtrMap:
  case POC::VariadicPtrRemoveMap:
    break;
  default:
    if (!llvm::all_of(operands, [&](auto operand) {
          return operand.getType() == operands.front().getType();
        }))
      return emitError() << "operand type mismatch";
  }

  // Check invariants on the expression.
  switch (opcode) {
  case POC::Add:
  case POC::Mul:
  case POC::MulNuw:
  case POC::And:
  case POC::Or:
  case POC::Xor:
  case POC::Max:
  case POC::Min:
    if (operands.empty())
      return emitError() << stringifyEnum(opcode)
                         << " operator must have at least one operand";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    // Check the types that are supported.
    if (type.isIntOrIndex())
      break; // Index and fixed-width integer types supported for all of these.
    return emitError() << "operator requires an index or integer type";
    break;
  // Binary expressions.
  case POC::Shl:
  case POC::Shr:
  case POC::Div:
  case POC::Mod:
    if (operands.size() != 2)
      return emitError() << stringifyEnum(opcode) << " must have two operands";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    if (!operands[0].getType().isIntOrIndex())
      return emitError() << "operator requires an index or integer type";
    break;
  case POC::EQ:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  case POC::LT:
  case POC::LE:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";

    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";

    // Relational operations only work on index types.
    if (!operands[0].getType().isIntOrIndex())
      return emitError() << "relational comparisons only allowed on index or "
                            "integer values";
    break;
  case POC::CurrentTarget:
    if (!operands.empty())
      return emitError() << "'current_target' expected no operands";
    if (!llvm::isa<TargetType>(type))
      return emitError() << "'current_target' must return a target type";
    break;
  case POC::TargetHasFeature:
  case POC::TargetGetField:
    if (operands.size() != 2)
      return emitError() << "target_get_field must have two operands";
    if (!llvm::isa<TargetType>(operands[0].getType()))
      return emitError() << "target_get_field operand 0 must be a target type";
    if (!llvm::isa<StringType>(operands[1].getType()))
      return emitError() << "target_get_field operand 1 must be a string type";
    break;
  case POC::In:
    if (operands.empty())
      return emitError() << "operator requires at least one operand";
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  case POC::GetSizeOf:
  case POC::GetAlignOf:
    if (operands.size() != 2) {
      return emitError() << stringifyEnum(opcode)
                         << " operator requires two operands";
    }
    if (!isTypeExpr(operands.front())) {
      return emitError() << stringifyEnum(opcode)
                         << " operand 0 should be a type expression";
    }
    if (!::isa<TargetType>(operands[1].getType())) {
      return emitError() << stringifyEnum(opcode)
                         << " operand 1 should be a !kgen.target";
    }
    if (!::isa<IntLiteralType, IndexType>(type)) {
      return emitError() << stringifyEnum(opcode)
                         << " should return an index or !kgen.int_literal";
    }
    break;
  case POC::BindSignature: {
    // It's possible that a function's specialized signature is more specific
    // than KGEN can determine using a `ParameterEvaluator`. In particular,
    // types need to be allowed to vary when parameter expression nodes rooted
    // at 'apply' operators are allowed to differ.
    break;
  }
  case POC::Apply:
    if (failed(verifyApply(operands, type, emitError)))
      return failure();
    break;
  case POC::ApplyResultSlot:
    if (failed(verifyApplyResultSlot(operands, type, emitError)))
      return failure();
    break;
  case POC::Rebind:
    if (operands.size() != 1)
      return emitError() << "'rebind' expects one operand";
    break;
  case POC::VariadicGet: {
    if (operands.size() != 2)
      return emitError() << "'variadic_get' expected two operands";
    auto variadicType = ::dyn_cast<VariadicType>(operands.front().getType());
    if (!variadicType)
      return emitError()
             << "'variadic_get' expected first operand to be a variadic value";
    if (!::isa<IndexType>(operands.back().getType()))
      return emitError()
             << "'variadic_get' expected second operand to be an index";
    Type elType = variadicType.getElementType();
    if (type != elType)
      return emitError() << "'variadic_get' result type should be variadic "
                            "element type: expected "
                         << elType << " but got " << type;
    break;
  }
  case POC::Cond:
    if (operands.size() != 3)
      return emitError() << "conditional expressions must have three operands";
    if (!operands[0].getType().isInteger(1))
      return emitError() << "conditional expression operand 0 must be i1";
    if (operands[1].getType() != operands[2].getType())
      return emitError() << "conditional expression operands 1 and 2 must have "
                            "the same type";
    if (operands[1].getType() != type)
      return emitError() << "result type should match operands 1 and 2 types";
    break;
  case POC::GetEnv:
    if (operands.size() != 1 || !::isa<StringType>(operands.front().getType()))
      return emitError() << "'get_env' expects one string-typed operand";
    if (auto intType = ::dyn_cast<IntegerType>(type)) {
      if (!intType.isSignlessInteger(1))
        return emitError() << "'get_env' must return index, i1, or string";
    } else if (!::isa<IndexType, StringType>(type)) {
      return emitError() << "'get_env' must return index, i1, or string";
    }
    break;
  case POC::CompileAssembly: {
    if (operands.size() != 4)
      return emitError() << "'compile_assembly' requires 4 operands";
    if (!::isa<TargetType>(operands.front().getType()))
      return emitError()
             << "'compile_assembly' first operand should be a target type";
    if (!::isa<IndexType>(operands[1].getType()))
      return emitError() << "'compile_assembly' second operand should be "
                            "either asm or llvm keyword";
    if (!operands[2].getType().isInteger(1))
      return emitError() << "'compile_assembly' third operand should be an i1";
    if (!::isa<IntegerAttr>(operands[2]))
      return emitError()
             << "'compile_assembly' fourth operand must be a constant";
    break;
  }
  case POC::GetLinkageName:
    if (operands.size() != 2)
      return emitError() << "'get_linkage_name' requires 2 operands";
    if (!::isa<TargetType>(operands.front().getType()))
      return emitError()
             << "'get_linkage_name' first operand should be a target type";
    break;
  case POC::GetTypeMethod:
    return verifyGetTypeMethod(operands, type, emitError);
  case POC::PtrBitcast:
    if (operands.size() != 1)
      return emitError() << "'ptr_bitcast' expects one operand";
    if (!::isa<PointerType>(type) ||
        !::isa<PointerType>(operands.front().getType()))
      return emitError() << "'ptr_bitcast' requires operand and result types "
                            "to both be pointers";
    break;
  case POC::LoadFromMem:
    if (operands.size() != 1)
      return emitError() << "'load_from_mem' expects one operand";
    break;
  case POC::VariadicPtrMap:
    return verifyVariadicPtrMap(operands, type, emitError);
  case POC::VariadicPtrRemoveMap:
    return verifyVariadicPtrRemoveMap(operands, type, emitError);
  }
  return success();
}

/// If the specified attribute is a ParamOperatorAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamOperatorAttr dyn_castPE(POC opcode, Attribute value) {
  if (auto expr = dyn_cast<ParamOperatorAttr>(value))
    if (expr.getOpcode() == opcode)
      return expr;
  return {};
}

/// Treat `index` and signed integers as signed. Treat signless and unsigned
/// integers as unsigned.
static bool isSignedIntType(Type type) {
  return type.isIndex() || type.isSignedInteger();
}

/// Given a function_ref from `(APInt,APInt)->T` and two APInt's, compute the
/// result value T and return it.
///
/// Note that this function has special behavior when 'valueTy' (the MLIR type
/// of the two operand values) is 'index' type. In this case, it does extra work
/// to make sure that a 32-bit and 64-bit target will compute the same result
/// using the same approach as the index dialect.  If they differ, this refuses
/// to fold the operation, returning a null IntegerAttr.
template <typename ResultTy>
static IntegerAttr foldBinaryValues(
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &unsignedCalculateFn,
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &signedCalculateFn,
    const APInt &lhs, const APInt &rhs, Type valueTy, Type resultTy = {}) {
  const auto &calculateFn =
      isSignedIntType(valueTy) ? signedCalculateFn : unsignedCalculateFn;

  // Clients can specify resultTy if it differs from valueTy (e.g. for
  // compares), but not specifying it defaults to the result being the same type
  // as the operands.
  if (!resultTy)
    resultTy = valueTy;

  auto result1 = calculateFn(lhs, rhs);
  if (!llvm::isa<IndexType>(valueTy))
    return IntegerAttr::get(resultTy, result1);

  // If this is an index computation, then we just did the 64-bit computation,
  // see what would happen on a 32-bit host.
  assert(lhs.getBitWidth() == 64);

  // We require that the computation satisfy the invariant that:
  //   trunc(f(a, b)) = f(trunc(a), trunc(b))
  auto result2 = calculateFn(lhs.trunc(32), rhs.trunc(32));

  // If not bool result (e.g. a compare), truncate the LHS for our check.
  auto result1test = result1;
  if constexpr (!std::is_same_v<bool, ResultTy>)
    result1test = result1.trunc(result2.getBitWidth());

  // We can use the full 64-bit folded result if they match, otherwise leave
  // unfolded.
  return result1test == result2 ? IntegerAttr::get(resultTy, result1)
                                : IntegerAttr();
}

/// Duplicate the operands in-place for ops like `min` and `max`.
static void deduplicateOperands(SmallVectorImpl<TypedAttr> &operands) {
  llvm::SetVector<TypedAttr, SmallVector<TypedAttr>, SmallPtrSet<Attribute, 4>>
      uniqueOperands(operands.begin(), operands.end());
  operands = uniqueOperands.takeVector();
}

/// Given a fully associative variadic integer operation, constant fold any
/// constant operands and move them to the right.  If the whole expression is
/// constant, then return that, otherwise update the operands list.
static Attribute simplifyAssocOp(
    POC opcode, SmallVectorImpl<TypedAttr> &operands,
    llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedFn,
    llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn = {},
    llvm::function_ref<bool(const APInt &)> identityConstantFn = {},
    llvm::function_ref<bool(const APInt &)> destructiveConstantFn = {},
    bool shouldDeduplicateOperands = false) {
  auto type = operands[0].getType();
  if (operands.size() == 1)
    return operands[0];

  // Flatten any of the same operation into the operand list:
  // `(add x, (add y, z))` => `(add x, y, z)`.
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto subexpr = dyn_castPE(opcode, operands[i])) {
      operands[i] = operands.back();
      operands.pop_back();
      --e;
      --i;
      operands.append(subexpr.getOperands().begin(),
                      subexpr.getOperands().end());
    }
  }

  // If allowed, deduplicate operands after flattening
  if (shouldDeduplicateOperands)
    deduplicateOperands(operands);

  // Impose an ordering on the operands, pushing subexpressions to the left and
  // constants to the right, with ParamRefs in the middle - but predictably
  // ordered w.r.t. each other.
  llvm::stable_sort(operands, ParameterAttr::compare);

  // Merge any constants, they will appear at the back of the operand list now.
  if (llvm::isa<IntegerAttr>(operands.back())) {
    while (operands.size() >= 2 &&
           llvm::isa<IntegerAttr>(operands[operands.size() - 2])) {
      APInt c1 =
          llvm::cast<IntegerAttr>(operands[operands.size() - 2]).getValue();
      APInt c2 = llvm::cast<IntegerAttr>(operands.back()).getValue();
      if (auto resultConstant = foldBinaryValues(
              unsignedFn, signedFn ? signedFn : unsignedFn, c1, c2, type)) {
        operands.pop_back();
        operands.pop_back();
        operands.push_back(resultConstant);
      } else {
        // If we couldn't fold the two values, bail.
        break;
      }
    }

    auto resultCst = llvm::cast<IntegerAttr>(operands.back());

    // If the resulting constant is the destructive constant (e.g. `x*0`), then
    // return it.
    if (destructiveConstantFn && destructiveConstantFn(resultCst.getValue()))
      return resultCst;

    // Remove the constant back to our operand list if it is the identity
    // constant for this operator (e.g. `x*1`) and there are other operands.
    if (identityConstantFn && identityConstantFn(resultCst.getValue()) &&
        operands.size() != 1)
      operands.pop_back();
  }

  return operands.size() == 1 ? operands[0] : Attribute();
}

/// Analyze an operand to an add.  If it is a multiplication by a constant (e.g.
/// `(a*b*42)` then split it into the non-constant and the constant portions
/// (e.g. `a*b` and `42`).  Otherwise return the operand as the first value and
/// null as the second (standin for "multiplication by 1").
static std::pair<TypedAttr, TypedAttr> decomposeAddend(TypedAttr operand) {
  auto mul = dyn_cast<ParamOperatorAttr>(operand);
  // NOTE we are bascially converting the "looser" MulNuw with undef behavior
  // back to the tighter Mul with defined behavior on overflow. This allows us
  // to fold things like `add(mul(X, 2), mul_nuw(X, -1))`
  if (mul && llvm::is_contained({POC::MulNuw, POC::Mul}, mul.getOpcode())) {
    if (auto cst = dyn_cast<IntegerAttr>(mul.getOperands().back())) {
      auto nonCst =
          ParamOperatorAttr::get(POC::Mul, mul.getOperands().drop_back());
      return {nonCst, cst};
    }
  }
  return {operand, TypedAttr()};
}

static Attribute getOneOfType(Type type) {
  size_t width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return IntegerAttr::get(type, APInt(width, 1));
}

static Attribute simplifyAdd(SmallVectorImpl<TypedAttr> &operands) {
  if (auto result = simplifyAssocOp(
          POC::Add, operands, [](auto a, auto b) { return a + b; }, {},
          /*identityCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // Canonicalize the add by splitting all addends into their variable and
  // constant factors.
  SmallVector<std::pair<TypedAttr, TypedAttr>> decomposedOperands;
  llvm::SmallDenseSet<TypedAttr> nonConstantParts;
  for (auto &op : operands) {
    decomposedOperands.push_back(decomposeAddend(op));

    // Keep track of non-constant parts we've already seen.  If we see multiple
    // uses of the same value, then we can fold them together with a multiply.
    // This handles things like `(a+b+a)` => `(a*2 + b)` and `(a*2 + b + a)` =>
    // `(a*3 + b)`.
    if (!nonConstantParts.insert(decomposedOperands.back().first).second) {
      // The thing we multiply will be the common expression.
      TypedAttr mulOperand = decomposedOperands.back().first;

      // Find the index of the first occurrence.
      size_t i = 0;
      while (decomposedOperands[i].first != mulOperand)
        ++i;
      // Remove both occurrences from the operand list.
      operands.erase(operands.begin() + (&op - &operands[0]));
      operands.erase(operands.begin() + i);

      auto type = mulOperand.getType();
      auto c1 = decomposedOperands[i].second,
           c2 = decomposedOperands.back().second;
      // Fill in missing constant multiplicands with 1.
      if (!c1)
        c1 = cast<TypedAttr>(getOneOfType(type));
      if (!c2)
        c2 = cast<TypedAttr>(getOneOfType(type));
      // Re-add the "a"*(c1+c2) expression to the operand list and
      // re-canonicalize.
      auto constant = ParamOperatorAttr::get(POC::Add, c1, c2);
      auto mulCst = ParamOperatorAttr::get(POC::Mul, mulOperand, constant);
      operands.push_back(mulCst);
      return ParamOperatorAttr::get(POC::Add, operands);
    }
  }

  return {};
}

static Attribute simplifyGenericMul(SmallVectorImpl<TypedAttr> &operands,
                                    POC opcode) {
  if (auto result = simplifyAssocOp(
          opcode, operands, [](auto a, auto b) { return a * b; }, {},
          /*identityCst*/ [](auto cst) { return cst.isOne(); },
          /*destructiveCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // We always build a sum-of-products representation, so if we see an addition
  // as a subexpr, we need to pull it out: (a+b)*c*d ==> (a*c*d + b*c*d).
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto addSubExpr = dyn_castPE(POC::Add, operands[i])) {
      // Pull the `c*d` operands out - it is whatever operands remain after
      // removing the `(a+b)` term.
      operands.erase(operands.begin() + i);

      // Build each add operand.
      SmallVector<TypedAttr> addOperands;
      for (auto addOperand : addSubExpr.getOperands()) {
        operands.push_back(addOperand);
        addOperands.push_back(ParamOperatorAttr::get(opcode, operands));
        operands.pop_back();
      }
      // Canonicalize and form the add expression.
      return ParamOperatorAttr::get(POC::Add, addOperands);
    }
  }

  return {};
}

static Attribute simplifyAnd(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::And, operands, [](auto a, auto b) { return a & b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*destructiveCst*/ [](auto cst) { return cst.isZero(); });
}

static Attribute simplifyOr(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Or, operands, [](auto a, auto b) { return a | b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isZero(); },
      /*destructiveCst*/ [](auto cst) { return cst.isAllOnes(); });
}

static Attribute simplifyXor(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Xor, operands, [](auto a, auto b) { return a ^ b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isZero(); });
}

/// Returns true if the integer is at its max value.
static bool intIsMaxValue(Type type, const APInt &value) {
  return isSignedIntType(type) ? value.isMaxSignedValue() : value.isMaxValue();
}

/// Returns true if the integer is at its min value.
static bool intIsMinValue(Type type, const APInt &value) {
  return isSignedIntType(type) ? value.isMinSignedValue() : value.isMinValue();
}

static Attribute simplifyMax(SmallVectorImpl<TypedAttr> &operands) {
  Type type = operands.front().getType();
  Attribute maybeConstAttr = simplifyAssocOp(
      POC::Max, operands, llvm::APIntOps::umax, llvm::APIntOps::smax,
      [&](auto cst) { return intIsMinValue(type, cst); },
      [&](auto cst) { return intIsMaxValue(type, cst); },
      /*shouldDeduplicateOperands*/ true);
  if (maybeConstAttr)
    return maybeConstAttr;

  // Add folding rule: max(a*x, a*y) --> a*max(x, y)
  IntegerAttr commonFactor;
  for (TypedAttr operand : operands) {
    // Operand must be a product.
    auto mulAttr = dyn_castPE(POC::MulNuw, operand);
    if (!mulAttr)
      return {};

    // The product must end with a constant integer attribute, which (if
    // present) will be canonicalized to be in the back
    auto factor = dyn_cast<IntegerAttr>(mulAttr.getOperands().back());
    if (!factor)
      return {};

    if (!commonFactor) {
      commonFactor = factor;
      continue;
    }

    // Else we have a common factor from previous operand, the new factor must
    // match it.
    if (commonFactor != factor)
      return {};

    // At this point, the invariant is `operand` is a product that ends with a
    // constant integer, which matches for all previous `operand`s.
  }

  // New operands with the common factor dropped from the end of each product.
  SmallVector<TypedAttr> newOperands;
  for (TypedAttr operand : operands) {
    auto mulAttr = dyn_castPE(POC::MulNuw, operand);

    // If the product has the form `x * commonFactor`, the new operand is `x`.
    size_t numOperands = mulAttr.getNumOperands();
    if (numOperands == 2) {
      newOperands.push_back(mulAttr.getOperand(0));
    } else {
      auto newOperand = ParamOperatorAttr::get(
          mulAttr.getOpcode(), mulAttr.getOperands().slice(0, numOperands - 1));
      newOperands.push_back(newOperand);
    }
  }
  auto newMax = ParamOperatorAttr::get(POC::Max, newOperands);
  auto product = ParamOperatorAttr::get(POC::MulNuw, {newMax, commonFactor});
  return product;
}

static Attribute simplifyMin(SmallVectorImpl<TypedAttr> &operands) {
  Type type = operands.front().getType();
  return simplifyAssocOp(
      POC::Min, operands, llvm::APIntOps::umin, llvm::APIntOps::smin,
      [&](auto cst) { return intIsMaxValue(type, cst); },
      [&](auto cst) { return intIsMinValue(type, cst); },
      /*shouldDeduplicateOperands*/ true);
}

/// Given a binary function, if the two operands are known constant integers,
/// use the specified fold functions to compute the result.
static Attribute
foldBinaryOp(ArrayRef<TypedAttr> operands,
             llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedFn,
             llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn) {
  assert(operands.size() == 2 && "binary operator always has two operands");
  if (auto lhs = dyn_cast<IntegerAttr>(operands[0]))
    if (auto rhs = dyn_cast<IntegerAttr>(operands[1])) {
      if (auto resultConstant =
              foldBinaryValues(unsignedFn, signedFn, lhs.getValue(),
                               rhs.getValue(), lhs.getType()))
        return resultConstant;
    }
  return {};
}

/// Folds constants given a comparison function that returns bool.  The client
/// must handle signedness etc.
static IntegerAttr foldCompareOp(
    TypedAttr lhs, TypedAttr rhs,
    llvm::function_ref<bool(const APInt &, const APInt &)> unsignedCompareFn,
    llvm::function_ref<bool(const APInt &, const APInt &)> signedCompareFn =
        {}) {
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs))
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      if (auto resultConstant = foldBinaryValues(
              unsignedCompareFn,
              signedCompareFn ? signedCompareFn : unsignedCompareFn,
              lhsInt.getValue(), rhsInt.getValue(), lhsInt.getType(),
              IntegerType::get(rhs.getContext(), 1)))
        return resultConstant;
    }
  return {};
}

/// Compute the result of == for the two specified attributes, handling the
/// index truncation issue but otherwise relying on MLIR's canonicalization of
/// attributes to do the job for us.  Both operands may be null, and this
/// returns null if no folding is possible.
static IntegerAttr foldEquality(TypedAttr lhs, TypedAttr rhs) {
  // foldCompareOp handles 32-bit truncation of input values correctly.
  if (lhs.getType().isIndex() && isa<IntegerAttr>(lhs) && isa<IntegerAttr>(rhs))
    return foldCompareOp(lhs, rhs, [](auto a, auto b) { return a == b; });

  // If the values have pointer equality, we know they are equal.
  if (lhs == rhs)
    return BoolAttr::get(rhs.getContext(), true);

  // Otherwise, we can use pointer equality for the attributes we support that
  // are known to have agreeable widths.
  if (ParameterAttr::isSimpleConstant(lhs) &&
      ParameterAttr::isSimpleConstant(rhs))
    return BoolAttr::get(rhs.getContext(), lhs == rhs);

  // Otherwise can't fold something like "x == y".
  return {};
}

static Attribute simplifyShl(SmallVectorImpl<TypedAttr> &operands) {
  // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
  // add/mul canonicalization (also handles constant folding).
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1])) {
    // NOTE: This is correct even for index types because an overlong shift will
    // turn the result to zero.
    // FIXME: getOneBitSet asserts the shift amount should be in-range.  We need
    // to check this.
    auto rhsCst = APInt::getOneBitSet(rhs.getValue().getBitWidth(),
                                      rhs.getValue().getZExtValue());
    return ParamOperatorAttr::get(POC::Mul, operands[0],
                                  IntegerAttr::get(rhs.getType(), rhsCst));
  }
  return {};
}

static Attribute simplifyShr(SmallVectorImpl<TypedAttr> &operands) {
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isZero())
      return operands[0]; // `x >> 0 = x`.
  // TODO: 0 >> x, -1 >>> x

  // FIXME: Must care about high bits.
  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.lshr(b); },
      [](auto a, auto b) { return a.ashr(b); });
}

/// Tracks the operands of MulNuw in a form which allows easy simplification.
namespace {
struct DivOperandInfo {
  // tracks the occurrences of non-integral operands only, e.g. D1
  SmallDenseMap<TypedAttr, size_t> symOccurences;

  // tracks the coalesced constant terms, e.g. mul_nuw(5, 10, D1)
  // this would be 5 * 10 = 50.
  int64_t constant = 1;

  Type attrType;

  // whether folding of `constant` leads to overflow on the current system
  // OR initialized with wrong attr (only support IntegerAttr and MulNuw)
  // OR when dealing with potential expressions which are sufficiently large
  //    as to differ in behavior on 32/64 bit systems
  bool isPoisoned = false;

  // Multiplies `constant` by `num`, checking overflow
  inline void updateConstant(IntegerAttr integerAttr) {
    int64_t num = integerAttr.getInt();
    int64_t new_constant = constant * num;
    if (num == 0) {
      // the power of 0 -- it's always going to be 0!
      constant = 0;

      // TODO: think of this more
      isPoisoned = false;
      return;
    }

    // poison if overflow on the current system OR would overflow on
    // 32 bit system for `index` types
    isPoisoned = isPoisoned || (new_constant / num != constant) ||
                 (new_constant > std::numeric_limits<int32_t>::max()) ||
                 (new_constant < std::numeric_limits<int32_t>::min());
    constant = new_constant;
  }

  /// Construct an Info object using a MulNuw operator, or constant IntegerAttr
  DivOperandInfo(TypedAttr attr) {
    constant = 1;
    attrType = attr.getType();

    if (auto constAttr = dyn_cast<IntegerAttr>(attr)) {
      updateConstant(constAttr);
      return;
    }

    if (auto mulAttr = dyn_castPE(POC::MulNuw, attr)) {
      for (TypedAttr numOpAttr : mulAttr.getOperands()) {
        if (auto constAttr = dyn_cast<IntegerAttr>(numOpAttr)) {
          updateConstant(constAttr);
        } else {
          ++symOccurences[numOpAttr];
        }
      }
      return;
    }

    if (auto declAttr = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
      ++symOccurences[declAttr];
      return;
    }

    // Not supported attr
    isPoisoned = true;
  }

  /// Create a new MulNuw expression from the info stored. If no symbolic
  /// variables are left, return an IntegerAttr, else return a MulNuw
  TypedAttr getExpression() {
    SmallVector<TypedAttr> operands;

    IntegerAttr constTerm = IntegerAttr::get(attrType, constant);

    operands.push_back(constTerm);
    for (auto [operand, occurrences] : symOccurences)
      for (size_t i = 0; i < occurrences; i++)
        operands.push_back(operand);

    if (operands.size() == 1) {
      // Implies `constant` only term
      return constTerm;
    }

    return ParamOperatorAttr::get(POC::MulNuw, operands);
  }

  /// Simplify terms in `numerator` and `denominator` assuming deriving terms
  /// are dividing each other. Mutates operands in place.
  static void simplifyDivInPlace(DivOperandInfo &numerator,
                                 DivOperandInfo &denominator) {
    SmallDenseMap<TypedAttr, size_t> &numeratorOperandOccurences =
        numerator.symOccurences;
    SmallDenseMap<TypedAttr, size_t> &denominatorOperandOccurences =
        denominator.symOccurences;

    // Emulate cancelling out shared operand(s) by decrementing their
    // occurrences. e.g., for
    //   `mul_nuw(D0, D2, D0)` with occurrence mapping `{ D0 : 2, D2 : 1 }`.
    //   `mul_nuw(D2, D0, D2)` with occurrence mapping `{ D0 : 1, D2 : 2 }`.
    // the new occurrence mappings are
    //   `{ D0 : 1, D2 : 0 }`.
    //   `{ D0 : 0, D2 : 1 }`.
    for (auto [numOpAttr, occurrences] : numeratorOperandOccurences) {
      if (size_t denomOccurrences =
              denominatorOperandOccurences.lookup(numOpAttr)) {
        size_t sharedOccurrences = std::min(occurrences, denomOccurrences);
        numeratorOperandOccurences[numOpAttr] -= sharedOccurrences;
        denominatorOperandOccurences[numOpAttr] -= sharedOccurrences;
      }
    }

    // Cancel out the constant terms
    if (numerator.constant == 0) {
      numerator.symOccurences.clear();
    }
    if (denominator.constant == 0) {
      denominator.symOccurences.clear();
    }
    if (numerator.constant != 0 && denominator.constant != 0) {
      // abs to keep signedness of constants
      int64_t gcd_term =
          std::abs(std::gcd(numerator.constant, denominator.constant));
      numerator.constant /= gcd_term;
      denominator.constant /= gcd_term;
    }
  }
};

} // namespace

/// Simplify division operands by cancelling out shared elements within
/// numerator and denominator products, e.g., `(a*b)/(b*b) --> a/b`
static void simplifyDivOperands(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr &numeratorAttr = operands[0];
  TypedAttr &denominatorAttr = operands[1];

  // Build mapping from each MulNuw op operand to the number of its occurrences,
  // e.g., for `mul_nuw(D0, 42, D0)`, we build the mapping `{ D0 : 2}, constant:
  // 42`
  DivOperandInfo numeratorInfo = DivOperandInfo(numeratorAttr);
  DivOperandInfo denominatorInfo = DivOperandInfo(denominatorAttr);

  // Poisoning: implies overflow in folding of constant @ precision of int64_t:
  //     e.g. mul_nuw(1e18, 1e18, D1) --> 1e90
  // Or numerator/denominator is is not a MulNuw or an IntegerAttr
  if (numeratorInfo.isPoisoned || denominatorInfo.isPoisoned)
    return;

  DivOperandInfo::simplifyDivInPlace(numeratorInfo, denominatorInfo);

  operands[0] = numeratorInfo.getExpression();
  operands[1] = denominatorInfo.getExpression();
}

static Attribute simplifyDiv(SmallVectorImpl<TypedAttr> &operands) {
  simplifyDivOperands(operands);

  // Implement support for identities like `x/1 = x`.
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isOne())
      return operands[0];

  // Note that division by 0 is undefined behavior.
  return foldBinaryOp(
      operands, [](auto a, auto b) { return b.isZero() ? b : a.udiv(b); },
      [](auto a, auto b) { return b.isZero() ? b : a.sdiv(b); });
}

static Attribute simplifyMod(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr lhs = operands[0];
  TypedAttr rhs = operands[1];

  // Check whether `x` is a multiple of `y`, only for simple cases.
  auto isMultipleOf = [](TypedAttr x, TypedAttr y) {
    if (x == y)
      return true;

    ArrayRef<TypedAttr> xProductOperands;
    if (auto mulAttr = dyn_castPE(POC::Mul, x))
      xProductOperands = mulAttr.getOperands();
    if (auto mulAttr = dyn_castPE(POC::MulNuw, x))
      xProductOperands = mulAttr.getOperands();
    return llvm::is_contained(xProductOperands, y);
  };

  // Add folding rule `(n * x) % x = 0` for `x` of integer type.
  if (lhs.getType().isIntOrIndex() && isMultipleOf(lhs, rhs))
    return IntegerAttr::get(rhs.getType(), 0);

  // Implement support for identities like `x%1 = 0`
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.urem(b); },
      [](auto a, auto b) { return a.srem(b); });
}

static Attribute simplifyEQ(SmallVectorImpl<TypedAttr> &operands) {
  // Make sure parameters are ordered correctly, which also matters if they
  // don't fold.
  llvm::stable_sort(operands, ParameterAttr::compare);

  return foldEquality(operands[0], operands[1]);
}

/// Simplify the < and <= operations.
static Attribute
simplifyRelationalCompare(POC opcode, SmallVectorImpl<TypedAttr> &operands) {
  auto rhs = dyn_cast<IntegerAttr>(operands[1]);
  auto lhs = dyn_cast<IntegerAttr>(operands[0]);

  if (rhs && !lhs) {
    // If this is a `(le x, RHS)` and RHS is a constant, canonicalize to `lt`.
    if (opcode == POC::LE) {
      if (intIsMaxValue(rhs.getType(), rhs.getValue())) // x <= 127 --> TRUE.
        return BoolAttr::get(rhs.getContext(), true);
      return ParamOperatorAttr::get(
          POC::LT, operands[0],
          IntegerAttr::get(rhs.getType(), rhs.getValue() + 1));
    }
    // If this is (x < MAXCST) canonicalize to (x != MAXCST).
    if (intIsMaxValue(rhs.getType(), rhs.getValue()))
      return ParamOperatorAttr::getNE(operands[0], rhs);
  }

  if (lhs && !rhs) {
    // (le cst, x) -> !(lt x, cst)
    if (opcode == POC::LE)
      return ParamOperatorAttr::getNot(
          ParamOperatorAttr::get(POC::LT, operands[1], operands[0]));
    // (lt cst, x) -> !(le x, cst)
    return ParamOperatorAttr::getNot(
        ParamOperatorAttr::get(POC::LE, operands[1], operands[0]));
  }

  if (opcode == POC::LT)
    return foldCompareOp(
        operands[0], operands[1], [](auto a, auto b) { return a.ult(b); },
        [](auto a, auto b) { return a.slt(b); });
  assert(opcode == POC::LE);
  return foldCompareOp(
      operands[0], operands[1], [](auto a, auto b) { return a.ule(b); },
      [](auto a, auto b) { return a.sle(b); });
}

static Attribute simplifyHasFeature(SmallVectorImpl<TypedAttr> &operands) {
  auto target = dyn_cast<TargetParamAttr>(operands[0]);
  auto feature = dyn_cast<StringAttr>(operands[1]);
  if (!target || !feature)
    return {};
  return Builder(target.getContext())
      .getBoolAttr(target.getTarget().hasFeature(feature));
}

static Attribute simplifyTargetGetField(SmallVectorImpl<TypedAttr> &operands,
                                        Type &resultType) {
  auto target = dyn_cast<TargetParamAttr>(operands[0]);
  auto field = dyn_cast<StringAttr>(operands[1]);
  if (!field)
    return {};

  Builder b(field.getContext());
  if (llvm::is_contained<StringRef>({"triple", "os", "arch", "endianness"},
                                    field))
    resultType = b.getType<StringType>();
  else
    resultType = b.getType<IntLiteralType>();

  if (!target)
    return {};

  if (field.getValue() == "triple") {
    return StringAttr::get(target.getTarget().getTriple().getTriple(),
                           resultType);
  }
  if (field.getValue() == "os")
    return StringAttr::get(target.getTarget().getOS(), resultType);
  if (field.getValue() == "arch")
    return StringAttr::get(target.getTarget().getArch(), resultType);
  if (field.getValue() == "simd_bit_width")
    return b.getAttr<IntLiteralAttr>(target.getTarget().getSimdBitWidth());
  if (field.getValue() == "endianness") {
    return StringAttr::get(
        target.getTarget().getTriple().isLittleEndian() ? "little" : "big",
        resultType);
  }
  return {};
}

/// Simplifies an `in` (also `in(:dtype`) operator.  We know the all the
/// operands have the same type.
static Attribute simplifyIn(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr lhs = operands[0];
  MutableArrayRef<TypedAttr> trailing =
      llvm::MutableArrayRef(operands).drop_front();

  Builder b(lhs.getContext());

  // If there are no trailing operands, fold to false.
  if (trailing.empty())
    return b.getBoolAttr(false);

  // If there is only one trailing operand, canonicalize to an `eq` operator.
  if (trailing.size() == 1)
    return ParamOperatorAttr::get(POC::EQ, operands);

  bool allKnownFalse = true;
  for (TypedAttr operand : trailing) {
    // Fold to true if a match was found by value.
    if (auto knownEq = foldEquality(lhs, operand)) {
      if (knownEq.getValue().isOne())
        return knownEq;
    } else if (lhs == operand) {
      // Fold to true if they match symbolically, like "x+1" and "x+1".
      return b.getBoolAttr(true);
    } else {
      // If this is a symbolic comparison like "x == 5", then we cannot fold the
      // non-containment case.
      allKnownFalse = false;
    }
  }

  // Ok, we know that LHS isn't known to equal any member of the set, but it or
  // they might be symbolic.  If we know for sure that LHS *isn't* equal to any
  // of the elements in the set then we can fold to false.
  if (allKnownFalse)
    return b.getBoolAttr(false);

  // Sort and unique the trailing operands.
  llvm::stable_sort(trailing, ParameterAttr::compare);
  SmallVector<TypedAttr> newOperands;
  newOperands.reserve(operands.size());
  newOperands.push_back(lhs);
  SmallPtrSet<Attribute, 4> seenTrailing;
  for (TypedAttr operand : trailing)
    if (seenTrailing.insert(operand).second)
      newOperands.push_back(operand);
  if (newOperands == operands)
    return {};
  return ParamOperatorAttr::get(POC::In, newOperands);
}

/// Simplifies a `get_sizeof` operator. Try to narrow the operand to a type
/// constant. If it does, query its data layout.
static Attribute simplifyGetSizeOf(SmallVectorImpl<TypedAttr> &operands,
                                   Type &resultType) {
  Builder b(operands[0].getContext());
  if (!resultType)
    resultType = b.getIndexType();

  auto typeCst = dyn_cast<TypeConstantAttr>(operands[0]);
  auto target = dyn_cast<TargetParamAttr>(operands[1]);
  if (!typeCst || !target)
    return {};
  std::optional<int64_t> size = DataLayoutInterface::getTypeStoreSize(
      target.getTarget(), typeCst.getMlirType());
  if (!size)
    return {};

  if (isa<IndexType>(resultType))
    return b.getIndexAttr(*size);
  return b.getAttr<IntLiteralAttr>(*size);
}

/// Simplifies a `get_alignof` operator. Try to narrow the operand to a type
/// constant. If it does, query its data layout.
static Attribute simplifyGetAlignOf(SmallVectorImpl<TypedAttr> &operands,
                                    Type &resultType) {
  Builder b(operands[0].getContext());
  if (!resultType)
    resultType = b.getIndexType();

  auto typeCst = dyn_cast<TypeConstantAttr>(operands[0]);
  auto target = dyn_cast<TargetParamAttr>(operands[1]);
  if (!typeCst || !target)
    return {};
  std::optional<int64_t> size = DataLayoutInterface::getTypeABIAlign(
      target.getTarget(), typeCst.getMlirType());
  if (!size)
    return {};

  if (isa<IndexType>(resultType))
    return b.getIndexAttr(*size);
  return b.getAttr<IntLiteralAttr>(*size);
}

static Attribute simplifyBindSignature(MLIRContext *ctx,
                                       ArrayRef<TypedAttr> operands,
                                       Type &resultType) {
  // If there is only a single operand, then nothing is bound.
  if (operands.size() == 1)
    return operands[0];

  // Otherwise, compute the result type if requested. If an error is producted,
  // just abort.
  if (!resultType) {
    auto resultSigOr =
        verifyBindSignature(operands, [ctx]() -> mlir::InFlightDiagnostic {
          return mlir::emitError(UnknownLoc::get(ctx));
        });
    if (failed(resultSigOr))
      llvm::report_fatal_error("invalid bind_signature operator");
    resultType = *resultSigOr;
  }

  auto processSignatureLike = [&](auto attr, auto cloneWith) {
    bool hasUnboundParameters = attr.getParamValues().empty();
    hasUnboundParameters |=
        llvm::any_of(attr.getParamValues(),
                     [](TypedAttr value) { return isa<UnboundAttr>(value); });
    assert(hasUnboundParameters &&
           "cannot have already bound all the input parameters, because we'd "
           "end up with a nongeneric signature that would fail verification");

    if (attr.getParamValues().empty())
      return cloneWith(operands.drop_front(), cast<SignatureType>(resultType));

    // We have to interleave the new values wherever there's an unbound thing
    // so we preserve the order. Drop the first operand because it's the
    // signature itself.
    SmallVector<TypedAttr> paramValues;
    auto operandIt = operands.begin() + 1;
    for (TypedAttr param : attr.getParamValues()) {
      // If we have this parameter already, we're good. otherwise, bind it to
      // the operand provided.
      if (!isa<UnboundAttr>(param))
        paramValues.push_back(param);
      else
        paramValues.push_back(*operandIt++);
    }
    assert(operandIt == operands.end() && "Didn't use all the operands?");

    return cloneWith(paramValues, cast<SignatureType>(resultType));
  };

  // If the actual operand is a SymbolConstantAttr operand, then we can simplify
  // the bind_signature by folding the parameter values into it directly.
  if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(operands.front())) {
    return processSignatureLike(
        symbolConstant,
        [&](ArrayRef<TypedAttr> paramValues, SignatureType type) {
          return SymbolConstantAttr::get(symbolConstant.getSymbol(),
                                         paramValues, type);
        });
  }

  return {};
}

static Attribute simplifyApply(ArrayRef<TypedAttr> operands, Type &resultType) {
  TypedAttr func = operands.front();
  operands = operands.drop_front();
  // Take the result type.
  resultType = upbindApplyResult(
      cast<SignatureType>(func.getType()).getValues().getResult(0));

  if (auto opExpr = dyn_cast<MLIROpAttr>(func)) {
    // Make the operation real by materializing it into a fake block.
    // HACK: Should we be materializing IR inside an attribute's constructor?
    // Maybe defer this to the interpreter.
    auto block = std::make_unique<Block>();
    SmallVector<Value> fakeOperands;
    auto loc = UnknownLoc::get(func.getContext());
    for (Type type : opExpr.getType().getArguments())
      fakeOperands.push_back(block->addArgument(type, loc));
    OwningOpRef<Operation *> op =
        Operation::create(loc, {opExpr.getName(), func.getContext()},
                          opExpr.getType().getResults(), fakeOperands,
                          opExpr.getAttrs(), /*properties=*/nullptr);
    block->push_back(*op);
    // Verify the operation. Fail to fold if the operation is invalid. Silence
    // the error, since there is no way to report it.
    mlir::ScopedDiagnosticHandler handler(
        func.getContext(),
        [&](Diagnostic &diag) -> LogicalResult { return success(); });
    if (failed(mlir::verify(*op)))
      return {};
    SmallVector<OpFoldResult> results;
    SmallVector<Attribute> attrs;
    llvm::append_range(attrs, operands);
    if (failed((*op)->fold(attrs, results)))
      return {};
    assert(results.size() == 1 && "expected one operation result");
    if (auto result = results.front().dyn_cast<Attribute>())
      return cast<TypedAttr>(result);
    // If the fold hook returned an operation, just look up the corresponding
    // input.
    return operands[cast<BlockArgument>(results.front().get<Value>())
                        .getArgNumber()];
  }

  return {};
}

static TypedAttr simplifyRebind(ArrayRef<TypedAttr> operands, Type resultType) {
  assert(resultType && "rebind requires a result type");
  TypedAttr input = operands.front();
  if (input.getType() == resultType)
    return input;
  // Fold rebinds of an unbound.
  if (isa<UnboundAttr>(input))
    return UnboundAttr::get(resultType);

  // Fold rebinds of a StructType. Unify metatypes so information is not lost.
  if (auto typeCst = dyn_cast<TypeConstantAttr>(input))
    return TypeConstantAttr::get(typeCst.getTypeValue(), typeCst.getMlirType(),
                                 resultType);
  return {};
}

static TypedAttr simplifyVariadicGet(ArrayRef<TypedAttr> operands,
                                     Type &resultType) {
  resultType = cast<VariadicType>(operands.front().getType()).getElementType();

  if (auto variadic = dyn_cast<VariadicAttr>(operands.front())) {
    auto index = dyn_cast<IntegerAttr>(operands.back());
    if (!index || index.getInt() < 0 ||
        index.getInt() >= static_cast<ssize_t>(variadic.getValues().size()))
      return {};
    return variadic.getValues()[index.getInt()];
  }

  return {};
}

// Returns the op with operands replaced with substitutions with actual values.
// substitutions are automatically added in conditionals with equals.
//
// E.g. given substitutions = {C: 3} calling this on
//  cond(eq(A == 2 and C == 3), f(A, C), g(A, C))
//
// Will become cond(eq(A == 2 and 3 == 3), f(A, 3), g(A, 3))
//
// We are able to apply the substitution `A == 2` for the "then" branch
// Note the substitution from the equal can only be applied to the then branch.
//
// This substitution and walking is done in a recursive manner. The depth
// of this recursion is bound by the initial `depth_left` parameter.
static TypedAttr cloneOperandsWithSubstitution(
    TypedAttr op, const DenseMap<TypedAttr, IntegerAttr> &substitutions,
    size_t depth_left) {
  if (substitutions.contains(op))
    return substitutions.at(op);

  if (depth_left <= 0)
    return op;

  auto opParamOperator = dyn_cast<ParamOperatorAttr>(op);
  if (!opParamOperator)
    return op;

  // Constrain to IntegerAttr substitutions for now
  SmallVector<TypedAttr> newOperands;
  bool hasChanged = false;
  for (TypedAttr oldOperand : opParamOperator.getOperands()) {
    TypedAttr newOperand = cloneOperandsWithSubstitution(
        oldOperand, substitutions, depth_left - 1);
    newOperands.push_back(newOperand);
    hasChanged = hasChanged || (newOperand != oldOperand);
  }

  // No changes
  if (!hasChanged)
    return opParamOperator;

  auto result =
      ParamOperatorAttr::get(opParamOperator.getOpcode(), newOperands);
  return result;
}

static TypedAttr simplifyCond(ArrayRef<TypedAttr> operands) {
  TypedAttr condAttr = operands[0];
  TypedAttr thenAttr = operands[1];
  TypedAttr elseAttr = operands[2];
  if (thenAttr == elseAttr)
    return thenAttr;

  // cond(A != B, A, B) == A
  //
  // But A != B is represented as Xor(A == B, true)
  if (auto xorAttr = dyn_castPE(POC::Xor, condAttr)) {
    auto eqAttr = dyn_castPE(POC::EQ, xorAttr.getOperand(0));
    auto intAttr = dyn_cast<IntegerAttr>(xorAttr.getOperand(1));
    if (eqAttr && intAttr && intAttr.getValue().isOne() &&
        eqAttr.getOperand(0) == thenAttr && eqAttr.getOperand(1) == elseAttr)
      return thenAttr;
  }

  // cond(A == B, B, A) == A
  if (auto eqAttr = dyn_castPE(POC::EQ, condAttr)) {
    auto lhsEq = eqAttr.getOperand(0);
    auto rhsEq = eqAttr.getOperand(1);
    if (thenAttr == rhsEq && elseAttr == lhsEq)
      return lhsEq;

    auto rhsEqAsIntegral = dyn_cast<IntegerAttr>(rhsEq);
    auto lhsEqAsIntegral = dyn_cast<IntegerAttr>(lhsEq);

    // If in form cond(A == 5, f(A, ...), ...)
    // Substitute all occurrences of A in the then branch with '5' up to
    // `MAX_RECURSION_DEPTH`
    const static size_t MAX_RECURSION_DEPTH = 3;
    if (rhsEqAsIntegral && !lhsEqAsIntegral) {
      DenseMap<TypedAttr, IntegerAttr> substitutions = {
          {lhsEq, rhsEqAsIntegral}};
      TypedAttr newThenAttr = cloneOperandsWithSubstitution(
          thenAttr, substitutions, MAX_RECURSION_DEPTH);
      if (newThenAttr != thenAttr)
        return ParamOperatorAttr::get(POC::Cond,
                                      {condAttr, newThenAttr, elseAttr});
    }
  }

  // cond(X, false, X) == X
  if (auto then = dyn_cast<IntegerAttr>(thenAttr))
    if (then.getValue().isZero() && condAttr == elseAttr)
      return thenAttr;

  auto c = dyn_cast<IntegerAttr>(condAttr);
  if (!c)
    return {};
  if (c.getValue().isOne())
    return thenAttr;
  if (c.getValue().isZero())
    return elseAttr;
  return {};
}

static TypedAttr simplifyGetTypeMethod(ArrayRef<TypedAttr> operands,
                                       Type resultType) {
  auto typeConstant = dyn_cast<TypeConstantAttr>(operands[0]);
  // typeConstant may actually be a parameter if this is called before
  // elaboration.  But after elaboration it should always be a TypeConstantAttr.
  if (!typeConstant)
    return {};
  VTableAttr vtable = typeConstant.getVTable();
  StringAttr targetName = cast<StringAttr>(operands[1]);
  SignatureType targetSignature = cast<SignatureType>(resultType);

  // Scan the vtable for a name + signature match, then the method is the
  // payload.
  for (VTableEntryAttr entry : vtable.getEntries()) {
    if (entry.getName() == targetName.getValue() &&
        entry.getMethod().getType() == targetSignature) {
      return entry.getMethod();
    }
  }
  return {};
}

static TypedAttr simplifyPtrBitcast(ArrayRef<TypedAttr> operands,
                                    Type resultType) {
  if (operands.front().getType() == resultType)
    return operands.front();
  if (auto ptr = dyn_cast<PointerAttr>(operands.front()))
    return PointerAttr::get(ptr.getAddr(), resultType);
  return {};
}

static TypedAttr simplifyLoadFromMem(ArrayRef<TypedAttr> operands,
                                     Type resultType) {
  // If we get a PointerAttr, then it must not be mapped to any persistent
  // memory. There is nothing we can ever do with it. Return a undef value.
  if (isa<PointerAttr>(operands.front()))
    return UnknownAttr::get(resultType);
  return {};
}

static TypedAttr simplifyVariadicPtrMap(TypedAttr variadicOperand,
                                        TypedAttr addrSpaceOperand,
                                        Type resultType) {
  // Fold a concrete variadic list of types.
  auto variadic = dyn_cast<VariadicAttr>(variadicOperand);
  if (!variadic)
    return {};

  auto resultEltType = cast<VariadicType>(resultType).getElementType();

  SmallVector<TypedAttr> results;
  // Map each type to PointerType of their type.
  for (auto elt : variadic.getValues()) {
    results.push_back(TypeConstantAttr::get(
        PointerType::get(ParamRefType::get(elt), addrSpaceOperand),
        resultEltType));
  }

  return VariadicAttr::get(results, cast<VariadicType>(resultType));
}

static TypedAttr simplifyVariadicPtrRemoveMap(TypedAttr variadicOperand,
                                              Type resultType) {
  // Fold a concrete variadic list of types.
  auto variadic = dyn_cast<VariadicAttr>(variadicOperand);
  if (!variadic)
    return {};

  auto resultEltType = cast<VariadicType>(resultType).getElementType();

  SmallVector<TypedAttr> results;
  // Map each type from a PointerType of the element type.
  for (auto elt : variadic.getValues()) {
    auto eltCst = dyn_cast<TypeConstantAttr>(elt);
    if (!eltCst || !isa<PointerType>(eltCst.getMlirType()))
      return {};

    results.push_back(TypeConstantAttr::get(
        cast<PointerType>(eltCst.getTypeValue()).getElementType(),
        cast<PointerType>(eltCst.getMlirType()).getElementType(),
        resultEltType));
  }
  return VariadicAttr::get(results, cast<VariadicType>(resultType));
}

/// Construct a parameter operator attribute, folding it if possible.
static TypedAttr getParamOperator(MLIRContext *ctx, POC opcode,
                                  ArrayRef<TypedAttr> operandsIn,
                                  Type resultType) {
  SmallVector<TypedAttr, 4> operands(operandsIn.begin(), operandsIn.end());

  // Verify and canonicalize parameter expressions.
  Attribute result;
  switch (opcode) {
  case POC::Add:
    result = simplifyAdd(operands);
    break;
  case POC::Mul:
    [[fallthrough]];
  case POC::MulNuw:
    result = simplifyGenericMul(operands, opcode);
    break;
  case POC::And:
    result = simplifyAnd(operands);
    break;
  case POC::Or:
    result = simplifyOr(operands);
    break;
  case POC::Xor:
    result = simplifyXor(operands);
    break;
  case POC::Max:
    result = simplifyMax(operands);
    break;
  case POC::Min:
    result = simplifyMin(operands);
    break;
  case POC::Shl:
    result = simplifyShl(operands);
    break;
  case POC::Shr:
    result = simplifyShr(operands);
    break;
  case POC::Div:
    result = simplifyDiv(operands);
    break;
  case POC::Mod:
    result = simplifyMod(operands);
    break;
  case POC::EQ:
    result = simplifyEQ(operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::LT:
  case POC::LE:
    result = simplifyRelationalCompare(opcode, operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::CurrentTarget:
    resultType = TargetType::get(ctx);
    break;
  case POC::TargetHasFeature:
    result = simplifyHasFeature(operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::TargetGetField:
    result = simplifyTargetGetField(operands, resultType);
    break;
  case POC::In:
    result = simplifyIn(operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::GetSizeOf:
    result = simplifyGetSizeOf(operands, resultType);
    break;
  case POC::GetAlignOf:
    result = simplifyGetAlignOf(operands, resultType);
    break;
  case POC::BindSignature:
    result = simplifyBindSignature(ctx, operands, resultType);
    break;
  case POC::Apply:
    result = simplifyApply(operands, resultType);
    break;
  case POC::ApplyResultSlot:
    result = {};
    break;
  case POC::Rebind:
    result = simplifyRebind(operands, resultType);
    break;
  case POC::VariadicGet:
    result = simplifyVariadicGet(operands, resultType);
    break;
  case POC::Cond:
    result = simplifyCond(operands);
    break;
  case POC::GetEnv:
    result = {};
    break;
  case POC::CompileAssembly:
    result = {};
    break;
  case POC::GetLinkageName:
    result = {};
    break;
  case POC::GetTypeMethod:
    result = simplifyGetTypeMethod(operands, resultType);
    break;
  case POC::PtrBitcast:
    result = simplifyPtrBitcast(operands, resultType);
    break;
  case POC::LoadFromMem:
    result = simplifyLoadFromMem(operands, resultType);
    break;
  case POC::VariadicPtrMap:
    assert(operands.size() == 2 && "variadic_ptr_map always has 2 operands");
    result = simplifyVariadicPtrMap(operands[0], operands[1], resultType);
    break;
  case POC::VariadicPtrRemoveMap:
    assert(operands.size() == 1 && "variadic_ptrremove_map has 1 operand");
    result = simplifyVariadicPtrRemoveMap(operands[0], resultType);
    break;
  }

  // If we folded to an operand, return it.
  if (result)
    return cast<TypedAttr>(result);

  return ParamOperatorAttr::Base::get(ctx, opcode, operands, resultType);
}

TypedAttr ParamOperatorAttr::get(MLIRContext *context, POC opcode,
                                 ArrayRef<TypedAttr> operandsIn, Type type) {
  return getParamOperator(context, opcode, operandsIn, type);
}

TypedAttr
ParamOperatorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, POC opcode,
                              ArrayRef<TypedAttr> operandsIn, Type type) {
  if (failed(verify(emitError, opcode, operandsIn, type)))
    return {};
  return get(context, opcode, operandsIn, type);
}

TypedAttr ParamOperatorAttr::get(POC opcode, ArrayRef<TypedAttr> operandsIn) {
  assert(!operandsIn.empty() && "Cannot have expr with no operands");
  // All operands must have the same type.  The result type is usually the
  // same as the operands, but is i1 for comparisons (overridden below).
  Type resultType;
  if (opcode == POC::Cond)
    resultType = operandsIn[1].getType();
  else if (opcode != POC::BindSignature && opcode != POC::GetSizeOf &&
           opcode != POC::GetAlignOf)
    resultType = operandsIn.front().getType();
  assert(llvm::is_contained(
             {POC::BindSignature, POC::Apply, POC::ApplyResultSlot,
              POC::TargetHasFeature, POC::TargetGetField, POC::GetSizeOf,
              POC::GetAlignOf, POC::VariadicGet, POC::GetEnv,
              POC::CompileAssembly, POC::GetLinkageName, POC::GetTypeMethod,
              POC::VariadicPtrMap, POC::VariadicPtrRemoveMap},
             opcode) ||
         llvm::all_of(operandsIn.drop_front(),
                      [&](auto op) { return op.getType() == resultType; }));

  return getParamOperator(operandsIn.front().getContext(), opcode, operandsIn,
                          resultType);
}

/// Return (not x) which is the same as (xor x, true).  The `operand` value
/// must have type `i1`.
TypedAttr ParamOperatorAttr::getNot(TypedAttr operand) {
  TypedAttr one = BoolAttr::get(operand.getContext(), true);
  return ParamOperatorAttr::get(POC::Xor, {operand, one});
}

/// Return (neg x) which is the same as (mul x, -1).  The `operand` value
/// must have `index` type.
TypedAttr ParamOperatorAttr::getNeg(TypedAttr operand) {
  IntegerAttr minusOne =
      IntegerAttr::get(IndexType::get(operand.getContext()), APInt(64, -1ULL));
  return ParamOperatorAttr::get(POC::Mul, operand, minusOne);
}

/// Return (x-y) which is the same as (add x, (neg y)).  The `operand` value
/// must have `index` type.
TypedAttr ParamOperatorAttr::getSub(TypedAttr lhs, TypedAttr rhs) {
  return get(POC::Add, lhs, getNeg(rhs));
}

/// Parameter operators are the basis of parameter expressions and are never
/// simple constants.
bool ParamOperatorAttr::isConstant() const { return false; }

/// Sort operators by opcode, then number of operands, then recursively sort by
/// operand values.
std::optional<bool> ParamOperatorAttr::isLessThan(Attribute rhs) const {
  auto op = llvm::dyn_cast<ParamOperatorAttr>(rhs);
  // Expressions are always to the left of non-expressions.
  if (!op)
    return true;

  // Sort by string value of the opcode.
  if (getOpcode() != op.getOpcode())
    return stringifyPOC(getOpcode()) < stringifyPOC(op.getOpcode());

  // If they are the same opcode, sort by arity. More complex expressions are to
  // the left.
  if (getNumOperands() != op.getNumOperands())
    return getNumOperands() > op.getNumOperands();

  // We know the two subexpressions are different (they'd otherwise be pointer
  // equivalent) so just go compare all of the elements.
  for (auto [lhs, rhs] : llvm::zip(getOperands(), op.getOperands())) {
    if (ParameterAttr::compare(lhs, rhs))
      return true;
    if (ParameterAttr::compare(rhs, lhs))
      return false;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MLIROpAttr
//===----------------------------------------------------------------------===//

LogicalResult MLIROpAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 StringAttr name, DictionaryAttr attrs,
                                 SignatureType type) {
  if (type.getNumResults() != 1)
    return emitError()
           << "operation parameter expression must return one result";
  if (!type.isConcrete())
    return emitError()
           << "operation parameter expression must be a concrete signature";
  return success();
}

/// An MLIR operation attribute is always a constant.
bool MLIROpAttr::isConstant() const { return true; }

TypedAttr KGEN::emitMLIROperationCall(
    StringRef opName,
    ArrayRef<std::pair<StringAttr (*)(mlir::OperationName), Attribute>> attrs,
    ArrayRef<TypedAttr> operands, Type resultType) {
  MLIRContext *ctx = resultType.getContext();
  mlir::OperationName name(opName, ctx);
  NamedAttrList attrList;
  for (auto [attrName, value] : attrs)
    attrList.append(attrName(name), value);
  SmallVector<Type> operandTypes;
  for (TypedAttr operand : operands)
    operandTypes.push_back(operand.getType());
  SmallVector<TypedAttr> applyOperands;
  applyOperands.push_back(
      MLIROpAttr::get(name.getIdentifier(), attrList.getDictionary(ctx),
                      SignatureType::get(ctx, operandTypes, resultType)));
  llvm::append_range(applyOperands, operands);
  return ParamOperatorAttr::get(POC::Apply, applyOperands);
}

//===----------------------------------------------------------------------===//
// CustomOpImplAttr
//===----------------------------------------------------------------------===//

CustomOpImplAttr CustomOpImplAttr::get(StringAttr opName,
                                       SymbolConstantAttr opImplementation,
                                       SymbolConstantAttr opCanonicalization) {
  return CustomOpImplAttr::get(opName.getContext(), opName, opImplementation,
                               opCanonicalization);
}

CustomOpImplAttr CustomOpImplAttr::get(StringRef opName,
                                       SymbolConstantAttr opImplementation,
                                       SymbolConstantAttr opCanonicalization) {
  MLIRContext *context = opImplementation.getContext();
  auto opNameAttr = StringAttr::get(context, opName);
  return CustomOpImplAttr::get(context, opNameAttr, opImplementation,
                               opCanonicalization);
}

//===----------------------------------------------------------------------===//
// CustomOpImplArray
//===----------------------------------------------------------------------===//

/// Compare the names of two custom op implementations.
/// This function is used for sorting `CustomOpImplArrayAttr`.
/// Arguments are passed as `const *` to satisfy `llvm::array_pod_start`
/// signature.
static int compareOpImplNames(const CustomOpImplAttr *lhs,
                              const CustomOpImplAttr *rhs) {
  if (*lhs == *rhs)
    return 0;
  return lhs->getOpName().compare(rhs->getOpName());
}

/// Sort an array of `CustomOpImplAttr` in a given empty storage.
static bool opImplArrayAttrSort(ArrayRef<CustomOpImplAttr> value,
                                SmallVectorImpl<CustomOpImplAttr> &storage) {
  // Common case
  if (value.empty())
    return false;

  storage.assign(value.begin(), value.end());

  // Only sort if necessary.
  bool isSorted =
      llvm::is_sorted(value, [](CustomOpImplAttr l, CustomOpImplAttr r) {
        return compareOpImplNames(&l, &r);
      });
  if (!isSorted)
    llvm::array_pod_sort(storage.begin(), storage.end(), compareOpImplNames);
  return !isSorted;
}

LogicalResult
CustomOpImplArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<CustomOpImplAttr> opImpls) {
  for (int i = 0, e = (int)opImpls.size() - 1; i < e; i++)
    if (opImpls[i].getOpName() == opImpls[i + 1].getOpName())
      return emitError() << opImpls[i].getOpName() << " is defined twice";
  return success();
}

CustomOpImplArrayAttr
CustomOpImplArrayAttr::get(mlir::MLIRContext *ctx,
                           ArrayRef<CustomOpImplAttr> opImpls) {
  SmallVector<CustomOpImplAttr> sortedOpImpls;
  opImplArrayAttrSort(opImpls, sortedOpImpls);
  return Base::get(ctx, sortedOpImpls);
}

CustomOpImplArrayAttr
CustomOpImplArrayAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  mlir::MLIRContext *ctx,
                                  ArrayRef<CustomOpImplAttr> opImpls) {
  if (failed(verify(emitError, opImpls)))
    return {};

  SmallVector<CustomOpImplAttr> sortedOpImpls;
  opImplArrayAttrSort(opImpls, sortedOpImpls);
  return Base::get(ctx, sortedOpImpls);
}

CustomOpImplAttr CustomOpImplArrayAttr::getOpImpl(StringAttr opName) {
  ArrayRef<CustomOpImplAttr> value = getValue();

  // We write our own binary search here, as both std and llvm are assuming
  // that we are searching knowing
  const CustomOpImplAttr *attr = llvm::lower_bound(
      value, opName, [](CustomOpImplAttr lhs, StringAttr opName) {
        return lhs.getOpName().compare(opName) < 0;
      });
  if (attr == value.end() || attr->getOpName() != opName)
    return {};
  return *attr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
