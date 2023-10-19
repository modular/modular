//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
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
// Pretty Type Parsing and Printing utilities
//===----------------------------------------------------------------------===//

template <typename TypeT>
static ParseResult parsePrettyTypeImpl(AsmParser &p, TypedAttr &typeExpr) {
  Type type = TypeT::parse(p);
  if (!type)
    return failure();
  typeExpr = TypeConstantAttr::get(type);
  return success();
}

static Type parseScalarType(AsmParser &p) {
  TypedAttr resultDType;

  // Parse literal '<' + dtype + literal '>'
  if (p.parseLess() || failed(parseDTypeParamValue(p, resultDType)) ||
      p.parseGreater())
    return {};

  return SIMDType::get(1, resultDType);
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
    return emitError() << "expected type expression to be !kgen.mlirtype";
  return success();
}

std::optional<int64_t> POP::ArrayType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

Type POP::ArrayType::getElementAsType() const {
  TypedAttr eltType = getElementType();
  if (!eltType)
    return {};
  if (auto typeCst = dyn_cast_if_present<TypeConstantAttr>(eltType))
    return typeCst.getValue();
  assert(::isa<MLIRTypeType>(eltType.getType()));
  return ParamRefType::get(eltType);
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
  std::optional<int64_t> size = getResolvedSize();
  if (!size)
    return {};

  Type elementType = getElementAsType();
  std::optional<int64_t> elementAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, elementType);
  if (!elementAllocSize)
    return {};

  return *size * *elementAllocSize;
}

/// The alignment of the array is the alignment of the element type.
std::optional<int64_t>
POP::ArrayType::getTypeAlign(TargetInfoAttr target) const {
  Type elementType = getElementAsType();
  return DataLayoutInterface::getTypeABIAlign(target, elementType);
}

ErrorOrSuccess POP::ArrayType::writeTo(TypedAttr value, int64_t addr,
                                       InterpreterState &state) const {
  auto dl = getElementAsType().cast<DataLayoutInterface>();
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

ErrorOr<TypedAttr> POP::ArrayType::readFrom(int64_t addr,
                                            InterpreterState &state) const {
  Type elemType = getElementAsType();
  auto dl = elemType.cast<DataLayoutInterface>();
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
    return llvm::divideCeil(target.getDataLayout().getPointerBitWidth() * *size,
                            CHAR_BIT);
  default:
    return dtype->getSizeInBytes(*size);
  }
}

std::optional<int64_t> SIMDType::getTypeAlign(TargetInfoAttr target) const {
  if (std::optional<int64_t> size = getTypeSize(target))
    return llvm::PowerOf2Ceil(*size);
  return {};
}

ErrorOrSuccess SIMDType::writeTo(TypedAttr value, int64_t addr,
                                 InterpreterState &state) const {
  KGENDType dtype = *getResolvedDType();
  int64_t vecSize = *getTypeSize(state.getTarget());
  ErrorOr<void *> mem = state.getWritableMemory(addr, vecSize);
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

ErrorOr<TypedAttr> SIMDType::readFrom(int64_t addr,
                                      InterpreterState &state) const {
  DType dtype = *getResolvedDType();
  int64_t vecSize = *getTypeSize(state.getTarget());
  ErrorOr<const void *> mem = state.getReadableMemory(addr, vecSize);
  if (mem.isError())
    return mem.takeError();
  auto *data = reinterpret_cast<const uint8_t *>(*mem);
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
// VariantType
//===----------------------------------------------------------------------===//

VariantType VariantType::get(ArrayRef<Type> types) {
  assert(!types.empty());
  SmallVector<TypedAttr> typeExprs;
  for (Type type : types)
    typeExprs.push_back(TypeConstantAttr::get(type));
  return get(types.front().getContext(), typeExprs);
}

/// Return the number of types in the variant.
size_t VariantType::getNumTypes() { return getTypes().size(); }

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
/// content field is the biggest element size rounded up to the nearest
/// multiple of the content element type size, which is i64.
static std::optional<int64_t> computeVariantContentSize(VariantType type,
                                                        TargetInfoAttr target) {
  int64_t maxSize = 0;
  for (TypedAttr typeExpr : type.getTypes()) {
    auto typeCst = llvm::dyn_cast<ConcreteTypeConstantAttr>(typeExpr);
    if (!typeCst)
      return {};
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeAllocSize(target, typeCst.getValue());
    if (!typeSize)
      return {};
    maxSize = std::max(maxSize, *typeSize);
  }
  return llvm::alignTo(maxSize, *type.getTypeAlign(target));
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
  // Align to the content array element alignment. We don't expect the
  // discriminator to exceed it in size (at least a 32-bit integer).
  return llvm::alignTo(*contentSize + getVariantDiscrSize(*this),
                       *getTypeAlign(target));
}

std::optional<int64_t> VariantType::getTypeAlign(TargetInfoAttr target) const {
  // The alignment of the variant type is the alignment of the integer type
  // equal to the pointer width.
  // FIXME: This is incorrect but the LLVM lowering needs to be fixed.
  return target.getDataLayout().getIntegerABIAlign(
      target.getDataLayout().getPointerBitWidth());
}

ErrorOrSuccess VariantType::writeTo(TypedAttr value, int64_t addr,
                                    InterpreterState &state) const {
  // Just write the value to the address and then the discriminator.
  auto variant = ::cast<VariantAttr>(value);
  TypedAttr typeValue = variant.getValue();
  ErrorOrSuccess result = state.writeAttributeToMemory(addr, typeValue);
  if (result.isError())
    return result.takeError();
  addr += *computeVariantContentSize(*this, state.getTarget());

  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<void *> mem = state.getWritableMemory(addr, discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, variant.getIndex());
  llvm::StoreIntToMemory(discrVal, reinterpret_cast<uint8_t *>(*mem),
                         discrSize);
  return success();
}

ErrorOr<TypedAttr> VariantType::readFrom(int64_t addr,
                                         InterpreterState &state) const {
  // Read the discriminator first so we know what type to read.
  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<const void *> mem = state.getReadableMemory(
      addr + *computeVariantContentSize(*this, state.getTarget()), discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(discrVal, reinterpret_cast<const uint8_t *>(*mem),
                          discrSize);

  unsigned index = discrVal.getZExtValue();
  TypedAttr type = getTypes()[index];
  ErrorOr<TypedAttr> result = state.readAttributeFromMemory(
      addr, type.cast<ConcreteTypeConstantAttr>().getValue());
  if (result.isError())
    return result.takeError();
  return VariantAttr::get(result.takeValue(), index, *this);
}

//===----------------------------------------------------------------------===//
// CoroutineType
//===----------------------------------------------------------------------===//

std::optional<int64_t> CoroutineType::getTypeSize(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerSize();
}

std::optional<int64_t>
CoroutineType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

CoroutineType CoroutineType::get(SignatureType sig) {
  // Return a coroutine type whose result types match the signature type but
  // which inherits the `throws` bit.
  MLIRContext *ctx = sig.getContext();
  auto coroSig =
      SignatureType::get(FunctionType::get(ctx, {}, sig.getValueResults()), {},
                         {}, {}, FnEffects().setThrows(sig.isThrows()));
  return POP::CoroutineType::get(ctx, coroSig);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/POPDialect/POPTypes.cpp.inc"
      >();

  auto *dialect = getContext()->getOrLoadDialect<KGENDialect>();
  dialect->registerMnemonicType<ArrayType>();
  dialect->registerMnemonicType<StructType>();
  dialect->registerMnemonicType<VariantType>();

  dialect->registerKeywordParser("scalar", parseScalarType);
  dialect->registerPrettyType(
      "simd", &SIMDType::parse, TypeID::get<SIMDType>(),
      +[](AsmPrinter &p, Type type) {
        auto simd = cast<SIMDType>(type);
        if (simd.isScalar()) {
          p << "scalar<";
          printDTypeParamValue(p, simd.getDType());
          p << ">";
        } else {
          p << "simd";
          simd.print(p);
        }
      });
}

/// Parse a type registered to this dialect.
/// For most cases we rely on the default `generatedTypeParser`, but we have a
/// special handling for "scalar<t>", which is a syntactic sugar for
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
      << "unknown type `" << mnemonic << "` in dialect `" << getNamespace()
      << "`";
  return {};
}

/// Print a type registered to this dialect.
/// For most cases we rely on the default `generatedTypePrinter`, but we sugar
/// "simd<1, t>" to "scalar<t>".
void POPDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto simd = dyn_cast<SIMDType>(type); simd && simd.isScalar()) {
    p << "scalar<";
    printDTypeParamValue(p, simd.getDType());
    p << ">";
    return;
  }
  (void)generatedTypePrinter(type, p);
}
