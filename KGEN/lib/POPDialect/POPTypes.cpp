//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
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

LogicalResult
POP::ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                       TypedAttr size, TypedAttr elementType) {
  if (!size.getType().isa<IndexType>())
    return emitError() << "expected size expression to be index type";
  if (!elementType.getType().isa<MLIRTypeType>())
    return emitError() << "expected size expression to be !kgen.mlirtype";
  return success();
}

std::optional<int64_t> POP::ArrayType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

Type POP::ArrayType::getResolvedElementType() const {
  if (auto typeCst = dyn_cast_if_present<TypeConstantAttr>(getElementType()))
    return typeCst.getValue();
  return nullptr;
}

POP::ArrayType POP::ArrayType::get(TypedAttr size, TypedAttr elementType) {
  return get(size.getContext(), size, elementType);
}

POP::ArrayType POP::ArrayType::get(TypedAttr size, Type elementType) {
  return get(size.getContext(), size, TypeConstantAttr::get(elementType));
}

POP::ArrayType POP::ArrayType::get(int64_t size, Type elementType) {
  return get(Builder(elementType.getContext()).getIndexAttr(size), elementType);
}

POP::ArrayType POP::ArrayType::get(ValueRange elements) {
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
std::optional<int64_t>
POP::ArrayType::getTypeSize(TargetInfoAttr target) const {
  Type elementType = getResolvedElementType();
  std::optional<int64_t> size = getResolvedSize();
  if (!elementType || !size)
    return {};

  std::optional<int64_t> elementAlign =
      DataLayoutInterface::getTypeAlignInBytes(target, elementType);
  std::optional<int64_t> elementSize =
      DataLayoutInterface::getTypeSizeInBytes(target, elementType);
  if (!elementAlign || !elementSize)
    return {};

  return *size * llvm::alignTo(*elementSize, *elementAlign);
}

/// The alignment of the array is the alignment of the element type.
std::optional<int64_t>
POP::ArrayType::getTypeAlign(TargetInfoAttr target) const {
  Type elementType = getResolvedElementType();
  if (!elementType)
    return {};
  return DataLayoutInterface::getTypeAlignInBytes(target, elementType);
}

ErrorOrSuccess POP::ArrayType::writeTo(TypedAttr value, intptr_t addr,
                                       InterpreterState &state) const {
  auto dl = getResolvedElementType().cast<DataLayoutInterface>();
  // Store each element spaced apart by padding according to its alignment.
  int64_t offset = llvm::alignTo(*dl.getTypeSize(state.getTarget()),
                                 *dl.getTypeAlign(state.getTarget()));
  for (TypedAttr value : value.cast<POP::ArrayAttr>().getValues()) {
    ErrorOrSuccess result = state.writeAttributeToMemory(addr, value);
    if (result.isError())
      return result.takeError();
    addr += offset;
  }
  return success();
}

ErrorOr<TypedAttr> POP::ArrayType::readFrom(intptr_t addr,
                                            InterpreterState &state) const {
  Type elemType = getResolvedElementType();
  auto dl = getResolvedElementType().cast<DataLayoutInterface>();
  int64_t offset = llvm::alignTo(*dl.getTypeSize(state.getTarget()),
                                 *dl.getTypeAlign(state.getTarget()));
  SmallVector<TypedAttr> values;
  for (int64_t i = 0, e = *getResolvedSize(); i != e; ++i, addr += offset) {
    ErrorOr<TypedAttr> result = state.readAttributeFromMemory(addr, elemType);
    if (result.isError())
      return result.takeError();
    values.push_back(result.takeValue());
  }
  return POP::ArrayAttr::get(values, *this);
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

std::optional<int64_t> PointerType::getTypeSize(TargetInfoAttr target) const {
  return target.getPointerSize();
}

std::optional<int64_t> PointerType::getTypeAlign(TargetInfoAttr target) const {
  return target.getPointerSize();
}

ErrorOrSuccess PointerType::writeTo(TypedAttr value, intptr_t addr,
                                    InterpreterState &state) const {
  int64_t size = state.getTarget().getPointerSize();
  ErrorOr<void *> mem = state.getMemory(addr, size);
  if (mem.isError())
    return mem.takeError();
  // The pointer size of the target is variable.
  APInt intVal(size * CHAR_BIT, value.cast<PointerAttr>().getAddr());
  llvm::StoreIntToMemory(intVal, reinterpret_cast<uint8_t *>(*mem), size);
  return success();
}

ErrorOr<TypedAttr> PointerType::readFrom(intptr_t addr,
                                         InterpreterState &state) const {
  int64_t size = state.getTarget().getPointerSize();
  ErrorOr<void *> mem = state.getMemory(addr, size);
  if (mem.isError())
    return mem.takeError();
  APInt intVal(size * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(intVal, reinterpret_cast<uint8_t *>(*mem), size);
  return PointerAttr::get(intVal.getLimitedValue(), *this);
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

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

std::optional<KGENDType> SIMDType::getResolvedDType() const {
  if (auto dtypeAttr = llvm::dyn_cast<DTypeConstantAttr>(getDType()))
    return dtypeAttr.getDType();
  return {};
}

std::optional<int64_t> SIMDType::getResolvedSize() const {
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

std::optional<int64_t> SIMDType::getTypeSize(TargetInfoAttr target) const {
  std::optional<KGENDType> dtype = getResolvedDType();
  std::optional<int64_t> size = getResolvedSize();
  if (!dtype || !size)
    return {};

  switch (dtype->getValue()) {
  case KGENDType::address:
  case KGENDType::index:
    return target.getPointerSize() * *size;
  default:
    return dtype->getSizeInBytes(*size);
  }
}

std::optional<int64_t> SIMDType::getTypeAlign(TargetInfoAttr target) const {
  if (std::optional<int64_t> size = getTypeSize(target))
    return llvm::PowerOf2Ceil(*size);
  return {};
}

bool M::KGEN::POP::isSIMDSizeOneType(Type type) {
  if (auto simd = dyn_cast_or_null<POP::SIMDType>(type)) {
    auto resolvedSize = simd.getResolvedSize();
    return (resolvedSize && *resolvedSize == 1);
  }
  return false;
}

ErrorOrSuccess SIMDType::writeTo(TypedAttr value, intptr_t addr,
                                 InterpreterState &state) const {
  KGENDType dtype = *getResolvedDType();
  int64_t vecSize = *getTypeSize(state.getTarget());
  ErrorOr<void *> mem = state.getMemory(addr, vecSize);
  if (mem.isError())
    return mem.takeError();
  auto *data = reinterpret_cast<uint8_t *>(*mem);
  ArrayRef<DTypeValue> values = value.cast<SIMDAttr>().getValues();

  // Integer dtypes s/ui1/2/4 are densely packed. Handle them here.
  if (dtype.isInt()) {
    unsigned bitWidth = dtype.getIntegerWidthInBits();
    if (bitWidth < CHAR_BIT) {
      assert(CHAR_BIT % bitWidth == 0);
      for (unsigned i = 0, e = values.size(); i != e;) {
        APInt value(CHAR_BIT, 0);
        for (unsigned j = 0; j != CHAR_BIT && i != e; j += bitWidth, ++i)
          value |= values[i].getIntVal().zext(CHAR_BIT).shl(j);
        llvm::StoreIntToMemory(value, data++, 1);
      }
      return success();
    }
  }

  // Other dtypes are multiples of bytes.
  int64_t byteSize = vecSize / *getResolvedSize();
  for (const DTypeValue &value : values) {
    llvm::StoreIntToMemory(value.getData(), data, byteSize);
    data += byteSize;
  }
  return success();
}

ErrorOr<TypedAttr> SIMDType::readFrom(intptr_t addr,
                                      InterpreterState &state) const {
  DType dtype = *getResolvedDType();
  int64_t vecSize = *getTypeSize(state.getTarget());
  ErrorOr<void *> mem = state.getMemory(addr, vecSize);
  if (mem.isError())
    return mem.takeError();
  auto *data = reinterpret_cast<uint8_t *>(*mem);
  int64_t count = *getResolvedSize();

  // Integer dtypes s/ui1/2/4 are densely packed. Handle them here.
  if (dtype.isInt()) {
    unsigned bitWidth = dtype.getIntegerWidthInBits();
    if (bitWidth < CHAR_BIT) {
      assert(CHAR_BIT % bitWidth == 0);
      SmallVector<DTypeValue> values;
      for (unsigned i = 0; i != count;) {
        APInt value(CHAR_BIT, 0);
        llvm::LoadIntFromMemory(value, data++, 1);
        for (unsigned j = 0; j != CHAR_BIT && i != count; j += bitWidth, ++i)
          values.emplace_back(value.lshr(j).trunc(bitWidth), dtype);
      }
      return SIMDAttr::get(values, *this);
    }
  }

  // Other dtypes are multiples of bytes.
  int64_t byteSize = vecSize / *getResolvedSize();
  SmallVector<DTypeValue> values;
  APInt value(byteSize * CHAR_BIT, 0);
  for (unsigned i = 0; i != count; ++i) {
    llvm::LoadIntFromMemory(value, data + i * byteSize, byteSize);
    values.emplace_back(value, dtype);
  }
  return SIMDAttr::get(values, *this);
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

Type StructType::getConcreteElementType(unsigned i) const {
  return llvm::cast<ConcreteTypeConstantAttr>(getElementTypes()[i]).getValue();
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

std::optional<int64_t> StructType::getTypeSize(TargetInfoAttr target) const {
  SmallVector<Type> types;
  if (failed(resolveElementTypes(types)))
    return {};
  int64_t size = 0;
  int64_t strictest = 1;
  for (Type type : types) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeAlignInBytes(target, type);
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeSizeInBytes(target, type);
    if (!typeAlign || !typeSize)
      return {};
    size = llvm::alignTo(size, *typeAlign) + *typeSize;
    strictest = std::max(strictest, *typeAlign);
  }
  return llvm::alignTo(size, strictest);
}

std::optional<int64_t> StructType::getTypeAlign(TargetInfoAttr target) const {
  SmallVector<Type> types;
  if (failed(resolveElementTypes(types)))
    return {};
  int64_t strictest = 1;
  for (Type type : types) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeAlignInBytes(target, type);
    if (!typeAlign)
      return {};
    strictest = std::max(strictest, *typeAlign);
  }
  return strictest;
}

ErrorOrSuccess StructType::writeTo(TypedAttr value, intptr_t addr,
                                   InterpreterState &state) const {
  intptr_t offset = 0;
  for (TypedAttr value : value.cast<StructAttr>().getValues()) {
    auto dl = value.getType().cast<DataLayoutInterface>();
    // Store each element spaced apart by padding according to its alignment.
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    ErrorOrSuccess result = state.writeAttributeToMemory(addr + offset, value);
    if (result.isError())
      return result.takeError();
    offset += *dl.getTypeSize(state.getTarget());
  }
  return success();
}

ErrorOr<TypedAttr> StructType::readFrom(intptr_t addr,
                                        InterpreterState &state) const {
  SmallVector<Type> elTypes;
  (void)resolveElementTypes(elTypes);
  SmallVector<TypedAttr> values;
  intptr_t offset = 0;
  for (Type elType : elTypes) {
    auto dl = elType.cast<DataLayoutInterface>();
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    ErrorOr<TypedAttr> value =
        state.readAttributeFromMemory(addr + offset, elType);
    if (value.isError())
      return value.takeError();
    values.push_back(value.takeValue());
    offset += *dl.getTypeSize(state.getTarget());
  }
  return StructAttr::get(values, *this);
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

std::optional<int64_t> VariantType::getTypeIndex(Type type) const {
  for (auto [idx, variantType] : llvm::enumerate(getTypes()))
    if (ParamRefType::get(variantType) == type)
      return idx;
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

/// Compute the size in bytes of just the content section of a variant. The
/// content field is the biggest element size rounded up to the nearest multiple
/// of the pointer width.
static std::optional<int64_t> computeVariantContentSize(VariantType type,
                                                        TargetInfoAttr target) {
  uint64_t maxSize = 0;
  for (TypedAttr typeExpr : type.getTypes()) {
    auto typeCst = llvm::dyn_cast<ConcreteTypeConstantAttr>(typeExpr);
    if (!typeCst)
      return {};
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeSizeInBytes(target, typeCst.getValue());
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeAlignInBytes(target, typeCst.getValue());
    if (!typeSize || !typeAlign)
      return {};
    maxSize = std::max(maxSize, llvm::alignTo(*typeSize, *typeAlign));
  }
  return llvm::alignTo(maxSize, target.getPointerSize());
}

/// Get bitwidth of the integer used to represent the discriminator. The
/// discriminator field is the smallest integer type whose maximum value is
/// greater than the number of possible subtypes, but which is at least `i1`.
static int64_t getVariantDiscrSizeInBits(VariantType type) {
  return std::max(1u, llvm::Log2_32_Ceil(type.getTypes().size()));
}

/// Get the width of the integer used to represent the discriminator in bytes.
/// This returns at least 1, because the bitwidth of the discriminator is at
/// least 1.
static int64_t getVariantDiscrSize(VariantType type) {
  return llvm::divideCeil(getVariantDiscrSizeInBits(type), CHAR_BIT);
}

std::optional<int64_t> VariantType::getTypeSize(TargetInfoAttr target) const {
  // A variant is lowered to a struct that consists of a content field and a
  // discriminator field.
  std::optional<int64_t> contentSize = computeVariantContentSize(*this, target);
  if (!contentSize)
    return {};
  return llvm::alignTo(*contentSize + getVariantDiscrSize(*this),
                       target.getPointerSize());
}

std::optional<int64_t> VariantType::getTypeAlign(TargetInfoAttr target) const {
  // The alignment of the variant type is just the pointer width.
  // FIXME: This is incorrect but the LLVM lowering needs to be fixed.
  return target.getPointerSize();
}

ErrorOrSuccess VariantType::writeTo(TypedAttr value, intptr_t addr,
                                    InterpreterState &state) const {
  // Just write the value to the address and then the discriminator.
  TypedAttr typeValue = value.cast<VariantAttr>().getValue();
  ErrorOrSuccess result = state.writeAttributeToMemory(addr, typeValue);
  if (result.isError())
    return result.takeError();
  addr += *computeVariantContentSize(*this, state.getTarget());

  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<void *> mem = state.getMemory(addr, discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, *getTypeIndex(typeValue.getType()));
  llvm::StoreIntToMemory(discrVal, reinterpret_cast<uint8_t *>(*mem),
                         discrSize);
  return success();
}

ErrorOr<TypedAttr> VariantType::readFrom(intptr_t addr,
                                         InterpreterState &state) const {
  // Read the discriminator first so we know what type to read.
  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<void *> mem = state.getMemory(
      addr + *computeVariantContentSize(*this, state.getTarget()), discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(discrVal, reinterpret_cast<uint8_t *>(*mem),
                          discrSize);

  TypedAttr type = getTypes()[discrVal.getZExtValue()];
  ErrorOr<TypedAttr> result = state.readAttributeFromMemory(
      addr, type.cast<ConcreteTypeConstantAttr>().getValue());
  if (result.isError())
    return result.takeError();
  return VariantAttr::get(result.takeValue(), *this);
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
    SymbolRefAttr ref;
    auto refResult = p.parseOptionalAttribute(ref);
    if (refResult.has_value()) {
      if (failed(*refResult))
        return failure();

      FailureOr<ParamBindArrayAttr> paramValues;
      if (parseOptionalParamBindSpec(p, paramValues))
        return failure();
      Type result = DeclRefType::get(ref, *paramValues);
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
// ClosureType
//===----------------------------------------------------------------------===//

std::optional<int64_t> ClosureType::getTypeSize(TargetInfoAttr target) const {
  // FIXME: Implement this.
  llvm_unreachable("TODO: unimplemented");
}

std::optional<int64_t> ClosureType::getTypeAlign(TargetInfoAttr target) const {
  // FIXME: Implement this.
  llvm_unreachable("TODO: unimplemented");
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
