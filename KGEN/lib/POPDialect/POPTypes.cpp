//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/POPDialect/POPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                TypedAttr size, TypedAttr elementType) {
  if (!size.getType().isa<IndexType>())
    return emitError() << "expected size expression to be index type";
  if (!elementType.getType().isa<MLIRTypeType>())
    return emitError() << "expected size expression to be !kgen.mlirtype";
  return success();
}

Optional<int64_t> ArrayType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

Type ArrayType::getResolvedElementType() const {
  if (auto typeCst = dyn_cast_if_present<TypeConstantAttr>(getElementType()))
    return typeCst.getValue();
  return nullptr;
}

ArrayType ArrayType::get(TypedAttr size, TypedAttr elementType) {
  return get(size.getContext(), size, elementType);
}

ArrayType ArrayType::get(TypedAttr size, Type elementType) {
  return get(size.getContext(), size, TypeConstantAttr::get(elementType));
}

ArrayType ArrayType::get(int64_t size, Type elementType) {
  return get(Builder(elementType.getContext()).getIndexAttr(size), elementType);
}

ArrayType ArrayType::get(ValueRange elements) {
  assert(!elements.empty() && "expected non-empty elements");
  auto firstElement = elements.front();
  assert(llvm::all_of(elements,
                      [firstType = firstElement.getType()](Value v) {
                        return v.getType() == firstType;
                      }) &&
         "expected same element types");
  return get(elements.size(), firstElement.getType());
}

/// The size of the array is the number of elements times the size of each
/// aligned element.
Optional<int64_t> ArrayType::getTypeSize(TargetInfoAttr target) const {
  Type elementType = getResolvedElementType();
  Optional<int64_t> size = getResolvedSize();
  if (!elementType || !size)
    return {};

  Optional<int64_t> elementAlign =
      DataLayoutInterface::getTypeAlignInBytes(target, elementType);
  Optional<int64_t> elementSize =
      DataLayoutInterface::getTypeSizeInBytes(target, elementType);
  if (!elementAlign || !elementSize)
    return {};

  return *size * llvm::alignTo(*elementSize, *elementAlign);
}

/// The alignment of the array is the alignment of the element type.
Optional<int64_t> ArrayType::getTypeAlign(TargetInfoAttr target) const {
  Type elementType = getResolvedElementType();
  if (!elementType)
    return {};
  return DataLayoutInterface::getTypeAlignInBytes(target, elementType);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr type) {
  if (type && !type.getType().isa<MLIRTypeType>())
    return emitError() << "type parameter for pointer must be a !kgen.mlirtype";
  return success();
}

Type PointerType::getResolvedElementType() const {
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(getElementType()))
    return typeCst.getValue();
  return nullptr;
}

PointerType PointerType::get(TypedAttr elementType) {
  return PointerType::get(elementType.getContext(), elementType);
}

PointerType PointerType::get(Type elementType) {
  return PointerType::get(TypeConstantAttr::get(elementType));
}

Optional<int64_t> PointerType::getTypeSize(TargetInfoAttr target) const {
  return target.getPointerSize();
}

Optional<int64_t> PointerType::getTypeAlign(TargetInfoAttr target) const {
  return target.getPointerSize();
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

/// Get the size in bytes of a KGEN dtype.
static Optional<int64_t> getDTypeByteSize(TargetInfoAttr target,
                                          KGENDType dtype) {
  // KGEN dtypes.
  if (dtype.getValue() == KGENDType::address)
    return target.getPointerSize();

  // Generic DType.
  int64_t size = dtype.getWidthInBits();
  if (size == -1)
    return {};
  return llvm::divideCeil(size, CHAR_BIT);
}

/// Get the alignment in bytes of a KGEN dtype.
static Optional<int64_t> getDTypeByteAlign(TargetInfoAttr target,
                                           KGENDType dtype) {
  // KGEN dtypes.
  if (dtype.getValue() == KGENDType::address)
    return target.getPointerSize();

  // Generic DType.
  int64_t size = dtype.getWidthInBits();
  if (size == -1)
    return {};
  int64_t align = llvm::PowerOf2Ceil(llvm::divideCeil(size, CHAR_BIT));
  // Cap the alignment to the pointer size.
  return std::min(align, target.getPointerSize());
}

LogicalResult SIMDType::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr size, TypedAttr dtype) {
  if (!size || !dtype)
    return emitError() << "simd type requires size and dtype";
  if (!size.getType().isIndex())
    return emitError() << "size parameter for simd must have type `index`";
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for simd must be a !kgen.dtype";
  return success();
}

Optional<KGENDType> SIMDType::getResolvedDType() const {
  if (auto dtypeAttr = llvm::dyn_cast<DTypeConstantAttr>(getDType()))
    return dtypeAttr.getDType();
  return {};
}

Optional<int64_t> SIMDType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

SIMDType SIMDType::get(TypedAttr size, TypedAttr dtype) {
  return get(size.getContext(), size, dtype);
}

SIMDType SIMDType::get(int64_t size, TypedAttr dtype) {
  return get(Builder(dtype.getContext()).getIndexAttr(size), dtype);
}
SIMDType SIMDType::get(MLIRContext *ctx, int64_t size, KGENDType dtype) {
  return get(size, DTypeConstantAttr::get(ctx, dtype));
}

Optional<int64_t> SIMDType::getTypeSize(TargetInfoAttr target) const {
  Optional<KGENDType> dtype = getResolvedDType();
  Optional<int64_t> size = getResolvedSize();
  if (!dtype || !size)
    return {};

  Optional<int64_t> elSize = getDTypeByteSize(target, *dtype);
  Optional<int64_t> elAlign = getDTypeByteAlign(target, *dtype);
  if (!elSize || !elAlign)
    return {};
  // Take the next power of 2 of the SIMD size.
  return llvm::PowerOf2Ceil(*size) * llvm::alignTo(*elSize, *elAlign);
}

Optional<int64_t> SIMDType::getTypeAlign(TargetInfoAttr target) const {
  return getTypeSize(target);
}

bool M::KGEN::POP::isSIMDSizeOneType(Type type) {
  if (auto simd = dyn_cast_or_null<POP::SIMDType>(type)) {
    auto resolvedSize = simd.getResolvedSize();
    return (resolvedSize && *resolvedSize == 1);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<TypedAttr> elementTypes) {
  for (auto &elementType : llvm::enumerate(elementTypes)) {
    if (!elementType.value().getType().isa<MLIRTypeType>())
      return emitError() << "struct element type at index "
                         << elementType.index() << " is not a !kgen.mlirtype";
  }
  return success();
}

LogicalResult
StructType::resolveElementTypes(SmallVectorImpl<Type> &elementTypes) const {
  for (TypedAttr elementType : getElementTypes()) {
    if (auto type = llvm::dyn_cast<TypeConstantAttr>(elementType))
      elementTypes.push_back(type.getValue());
    else
      return failure();
  }
  return success();
}

SmallVector<Type> StructType::getParameterizedElementTypes() const {
  SmallVector<Type> elementTypes;
  elementTypes.reserve(getNumElements());
  for (TypedAttr elementType : getElementTypes())
    elementTypes.push_back(ParamRefType::get(elementType));
  return elementTypes;
}

StructType StructType::get(MLIRContext *ctx, ArrayRef<Type> elementTypes) {
  SmallVector<TypedAttr> elementTypeExprs;
  elementTypeExprs.reserve(elementTypes.size());
  for (Type elementType : elementTypes)
    elementTypeExprs.push_back(TypeConstantAttr::get(elementType));
  return get(ctx, elementTypeExprs);
}

StructType StructType::get(ArrayRef<Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least one element type");
  return get(elementTypes.front().getContext(), elementTypes);
}

Optional<int64_t> StructType::getTypeSize(TargetInfoAttr target) const {
  SmallVector<Type> types;
  if (failed(resolveElementTypes(types)))
    return {};
  int64_t size = 0;
  int64_t strictest = 1;
  for (Type type : types) {
    Optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeAlignInBytes(target, type);
    Optional<int64_t> typeSize =
        DataLayoutInterface::getTypeSizeInBytes(target, type);
    if (!typeAlign || !typeSize)
      return {};
    size = llvm::alignTo(size, *typeAlign) + *typeSize;
    strictest = std::max(strictest, *typeAlign);
  }
  return llvm::alignTo(size, strictest);
}

Optional<int64_t> StructType::getTypeAlign(TargetInfoAttr target) const {
  SmallVector<Type> types;
  if (failed(resolveElementTypes(types)))
    return {};
  int64_t strictest = 1;
  for (Type type : types) {
    Optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeAlignInBytes(target, type);
    if (!typeAlign)
      return {};
    strictest = std::max(strictest, *typeAlign);
  }
  return strictest;
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

/// Canonicalize the possible types of a variant. Deduplicate the types.
static SmallVector<TypedAttr>
canonicalizeVariantTypes(ArrayRef<TypedAttr> types) {
  SmallVector<TypedAttr> deduplicatedTypes;
  SmallPtrSet<Attribute, 4> seenTypes;
  deduplicatedTypes.reserve(types.size());
  for (TypedAttr type : types)
    if (seenTypes.insert(type).second)
      deduplicatedTypes.push_back(type);
  return deduplicatedTypes;
}

VariantType VariantType::get(MLIRContext *ctx, ArrayRef<TypedAttr> types) {
  return Base::get(ctx, canonicalizeVariantTypes(types));
}

VariantType VariantType::get(ArrayRef<Type> types) {
  assert(!types.empty());
  SmallVector<TypedAttr> typeExprs;
  for (Type type : types)
    typeExprs.push_back(TypeConstantAttr::get(type));
  return get(types.front().getContext(), typeExprs);
}

/// Return the number of types in the variant.
size_t VariantType::getNumTypes() { return getTypes().size(); }

Optional<int64_t> VariantType::getTypeIndex(Type type) const {
  for (auto &variantType : llvm::enumerate(getTypes()))
    if (ParamRefType::get(variantType.value()) == type)
      return variantType.index();
  return {};
}

SmallVector<Type> VariantType::getParameterizedElementTypes() const {
  SmallVector<Type> types;
  types.reserve(getTypes().size());
  for (TypedAttr type : getTypes())
    types.push_back(ParamRefType::get(type));
  return types;
}

Type VariantType::getType(unsigned index) {
  return ParamRefType::get(getTypes()[index]);
}

Optional<int64_t> VariantType::getTypeSize(TargetInfoAttr target) const {
  // FIXME: Implement this.
  llvm_unreachable("TODO: unimplemented");
}

Optional<int64_t> VariantType::getTypeAlign(TargetInfoAttr target) const {
  // FIXME: Implement this.
  llvm_unreachable("TODO: unimplemented");
}

//===----------------------------------------------------------------------===//
// Pretty Type Parsing and Printing
//===----------------------------------------------------------------------===//

template <typename TypeT>
static ParseResult parsePrettyTypeImpl(AsmParser &p,
                                       FailureOr<TypedAttr> &typeExpr) {
  Type type = TypeT::parse(p);
  if (!type)
    return failure();
  typeExpr = TypeConstantAttr::get(type);
  return success();
}

static Type parseScalarType(AsmParser &p) {
  FailureOr<TypedAttr> resultDType;

  // Parse literal '<' + dtype + literal '>'
  if (p.parseLess() || failed(parseDTypeParamValue(p, resultDType)) ||
      p.parseGreater())
    return {};

  return SIMDType::get(1, *resultDType);
}

static ParseResult parsePrettyScalarType(AsmParser &p,
                                         FailureOr<TypedAttr> &typeExpr) {
  Type t = parseScalarType(p);
  if (isa<SIMDType>(t)) {
    typeExpr = TypeConstantAttr::get(t);
    return success();
  }
  return failure();
}

/// Try to parse a pretty type or a standard MLIR type. A pretty type is a POP
/// type without the dialect prefix or a symbol reference.
ParseResult POP::parsePrettyType(AsmParser &p, FailureOr<TypedAttr> &typeExpr) {
  // Try to parse a symbol name as sugar for [LIT]DeclRefType.
  {
    StringAttr ref;
    if (succeeded(p.parseOptionalSymbolName(ref))) {
      // TODO: DeclRefType will eventually need @X::@Y::@Z.
      FailureOr<ParamBindArrayAttr> paramValues;
      if (parseOptionalParamBindSpec(p, paramValues))
        return failure();

      Type result = DeclRefType::get(FlatSymbolRefAttr::get(ref), *paramValues);
      typeExpr = TypeConstantAttr::get(result);
      return success();
    }
  }

  StringRef typeName;
  // Try to parse a keyword for a known POP type. Allow `dtype` for
  // `!kgen.dtype` as well. If this fails, defer to the parameter value parser.
  if (p.parseOptionalKeyword(
          &typeName,
          {ArrayType::getMnemonic(), PointerType::getMnemonic(),
           SIMDType::getMnemonic(), StructType::getMnemonic(),
           VariantType::getMnemonic(), DTypeType::getMnemonic(), "scalar"}))
    return parseTypeParamValue(p, typeExpr);

  if (typeName == ArrayType::getMnemonic())
    return parsePrettyTypeImpl<ArrayType>(p, typeExpr);
  if (typeName == PointerType::getMnemonic())
    return parsePrettyTypeImpl<PointerType>(p, typeExpr);
  if (typeName == SIMDType::getMnemonic())
    return parsePrettyTypeImpl<SIMDType>(p, typeExpr);
  if (typeName == StructType::getMnemonic())
    return parsePrettyTypeImpl<StructType>(p, typeExpr);
  if (typeName == VariantType::getMnemonic())
    return parsePrettyTypeImpl<VariantType>(p, typeExpr);
  if (typeName == "scalar")
    return parsePrettyScalarType(p, typeExpr);

  if (typeName == DTypeType::getMnemonic()) {
    typeExpr = TypeConstantAttr::get(DTypeType::get(p.getContext()));
    return success();
  }

  llvm_unreachable("unknown keyword");
}

/// Try to print a pretty type or a standard MLIR type. A pretty type is a POP
/// type without the dialect prefix.
void POP::printPrettyType(AsmPrinter &p, TypedAttr typeExpr) {
  // If this isn't a type constant, defer to the parameter value printer.
  auto typeCst = dyn_cast<TypeConstantAttr>(typeExpr);
  if (!typeCst)
    return printTypeParamValue(p, typeExpr);

  // Try to print on the known types. Fallback to the generic type printer
  // otherwise.
  llvm::TypeSwitch<Type>(typeCst.getValue())
      .Case<ArrayType, PointerType, StructType, VariantType>([&](auto popType) {
        p << decltype(popType)::getMnemonic();
        popType.print(p);
      })
      .Case([&](SIMDType popType) {
        if (isSIMDSizeOneType(popType)) {
          p << "scalar<";
          printDTypeParamValue(p, popType.getDType());
          p << ">";
          return;
        }
        p << SIMDType::getMnemonic();
        popType.print(p);
      })
      .Case([&](DeclRefType ref) {
        p << ref.getSymbol();
        printOptionalParamBindSpec(p, ref.getParamValues());
      })
      .Case([&](DTypeType) { p << DTypeType::getMnemonic(); })
      .Default([&](auto) { printTypeParamValue(p, typeExpr); });
}

static ParseResult
parseArrayOfPrettyTypes(AsmParser &p,
                        FailureOr<SmallVector<TypedAttr>> &values) {
  values.emplace();
  return p.parseCommaSeparatedList([&]() -> ParseResult {
    FailureOr<TypedAttr> value;
    if (failed(parsePrettyType(p, value)))
      return failure();
    values->push_back(*value);
    return success();
  });
}

static void printArrayOfPrettyTypes(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printPrettyType(p, value); });
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom parser and printer
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
/// For most cases we rely on the default `generatedTypeParser`, but we have a
/// special handling for "scalar<t>", which is a syntactix sugar for
/// "simd<1, t>".
Type POPDialect::parseType(DialectAsmParser &p) const {
  StringRef mnemonic;
  Type genType;
  mlir::OptionalParseResult parseResult =
      generatedTypeParser(p, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  if (mnemonic == "scalar")
    return parseScalarType(p);

  p.emitError(p.getCurrentLocation())
      << "unknown  type `" << mnemonic << "` in dialect `" << getNamespace()
      << "`";
  return {};
}

/// Print a type registered to this dialect.
/// For most cases we rely on the default `generatedTypePrinter`, but we sugar
/// "simd<1, t>" to "scalar<t>".
void POPDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (isSIMDSizeOneType(type)) {
    p << "scalar<";
    printDTypeParamValue(p, cast<SIMDType>(type).getDType());
    p << ">";
    return;
  }
  (void)generatedTypePrinter(type, p);
}
