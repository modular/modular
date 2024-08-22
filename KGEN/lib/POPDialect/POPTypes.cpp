//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Interpreter/InterpreterState.h"
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
                       TypedAttr size, Type elementType) {
  if (!llvm::isa<IndexType>(size.getType()))
    return emitError() << "expected size expression to be index type";
  return success();
}

std::optional<int64_t> POP::ArrayType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

POP::ArrayType POP::ArrayType::get(TypedAttr size, Type elementType) {
  MLIRContext *ctx = size.getContext();
  return get(ctx, size, elementType);
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

  Type elementType = getElementType();
  std::optional<int64_t> elementAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, elementType);
  if (!elementAllocSize)
    return {};

  return *size * *elementAllocSize;
}

/// The alignment of the array is the alignment of the element type.
std::optional<int64_t>
POP::ArrayType::getTypeAlign(TargetInfoAttr target) const {
  Type elementType = getElementType();
  return DataLayoutInterface::getTypeABIAlign(target, elementType);
}

ErrorOrSuccess POP::ArrayType::writeTo(TypedAttr value, int64_t addr,
                                       InterpreterState &state) const {
  auto dl = ::cast<DataLayoutInterface>(getElementType());
  // Store each element spaced apart by padding according to its alignment.
  int64_t offset = llvm::alignTo(*dl.getTypeSize(state.getTarget()),
                                 *dl.getTypeAlign(state.getTarget()));
  for (TypedAttr value : ::cast<POP::ArrayAttr>(value).getValues()) {
    ErrorOrSuccess result = state.writeAttributeToMemory(addr, value);
    if (result.isError())
      return result.takeError();
    addr += offset;
  }
  return success();
}

ErrorOr<TypedAttr> POP::ArrayType::readFrom(int64_t addr,
                                            InterpreterState &state) const {
  Type elemType = getElementType();
  auto dl = ::cast<DataLayoutInterface>(elemType);
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
// UnionType
//===----------------------------------------------------------------------===//

OptionalParseResult UnionType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return {};
  TypedAttr element;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseColonTypeParamValue(p, element) || p.parseRBrace())
    return failure();
  value =
      UnionAttr::getChecked([&] { return p.emitError(loc); }, element, *this);
  return mlir::success((bool)value);
}

LogicalResult UnionType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto attr = ::dyn_cast<UnionAttr>(value);
  if (!attr)
    return failure();
  p << '{';
  printColonTypeParamValue(p, attr.getValue());
  p << '}';
  return success();
}

std::optional<int64_t> UnionType::getTypeSize(TargetInfoAttr target) const {
  int64_t maxSize = 0;
  for (Type type : getTypes()) {
    std::optional<int64_t> size =
        DataLayoutInterface::getTypeAllocSize(target, type);
    if (!size)
      return {};
    maxSize = std::max(maxSize, *size);
  }
  return llvm::alignTo(maxSize, *getTypeAlign(target));
}

std::optional<int64_t> UnionType::getTypeAlign(TargetInfoAttr target) const {
  // The alignment of the union type is the alignment of the integer type
  // equal to the pointer width.
  // FIXME: This is incorrect but the LLVM lowering needs to be fixed.
  return target.getDataLayout().getIntegerABIAlign(
      target.getDataLayout().getPointerBitWidth());
}

ErrorOrSuccess UnionType::writeTo(TypedAttr value, int64_t addr,
                                  InterpreterState &state) const {
  return state.writeAttributeToMemory(addr,
                                      ::cast<UnionAttr>(value).getValue());
}

ErrorOr<TypedAttr> UnionType::readFrom(int64_t addr,
                                       InterpreterState &state) const {
  return Error("cannot read a union-typed value");
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
  if (!llvm::isa<DTypeType>(dtype.getType()))
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
    break;
  }
  ssize_t result = dtype->getSizeInBytes(*size);
  // Return zero size for invalid/nonmaterializable dtypes.
  if (result == -1)
    return 0;
  return result;
}

std::optional<int64_t> SIMDType::getTypeAlign(TargetInfoAttr target) const {
  if (std::optional<int64_t> size = getTypeSize(target))
    return std::max((int64_t)llvm::PowerOf2Ceil(*size), (int64_t)1);
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
  ArrayRef<DTypeValue> values = llvm::cast<SIMDAttr>(value).getValues();

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

  // Other dtypes are multiples of bytes in memory.
  int64_t bitWidth = dtype.getWidthInBits();
  int64_t byteSize = vecSize / *getResolvedSize();
  int64_t shiftBits = byteSize * CHAR_BIT - bitWidth;

  SmallVector<DTypeValue> values;
  APInt value(byteSize * CHAR_BIT, 0);
  for (unsigned i = 0; i != count; ++i) {
    llvm::LoadIntFromMemory(value, data + i * byteSize, byteSize);
    if (bitWidth == -1) {
      // dtype width unknown (e.g. address, index).
      values.emplace_back(value, dtype);
    } else {
      // For FloatTF32, right Shift 32 bit data by 13 bits and trunc to 19 bits;
      // other types, lshr and trunc are no ops.
      values.emplace_back(value.lshr(shiftBits).trunc(bitWidth), dtype);
    }
  }
  return SIMDAttr::get(values, *this);
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
  dialect->registerMnemonicType<UnionType>();

  dialect->registerKeywordParser("scalar", parseScalarType);
  dialect->registerPrettyType(
      "simd", &SIMDType::parse, mlir::TypeID::get<SIMDType>(),
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
