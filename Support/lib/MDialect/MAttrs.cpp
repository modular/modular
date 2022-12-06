//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrs.h"
#include "Support/Compiler/MLIRDenseAttrStorage.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <type_traits>

using namespace M;

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/MDialect/MAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// IntOrFPType
//===----------------------------------------------------------------------===//

namespace {
/// A helper class for manipulating float or integer element types.
class IntOrFPType : public Type {
public:
  using Type::Type;

  /// Support type inquiry.
  static bool classof(Type type) { return type.isIntOrFloat(); }

  /// Get the bitwidth.
  unsigned getWidth() const { return getIntOrFloatBitWidth(); }

  /// Get the type size in bytes rounded up to the nearest byte boundary.
  unsigned getNearestByteSize() const {
    return llvm::divideCeil(getWidth(), CHAR_BIT);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PrimitiveArray
//===----------------------------------------------------------------------===//

/// Require the element type to be a float or integer.
LogicalResult
PrimitiveArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           ArrayRef<uint8_t> data, Type elementType) {
  auto intOrFp = llvm::dyn_cast<IntOrFPType>(elementType);
  if (!intOrFp)
    return emitError() << "expected integer or float element type";
  // Disallow s/ui0.
  if (intOrFp.getWidth() == 0)
    return emitError() << "zero-width element type unsupported";
  // Sanity check the provided data.
  unsigned extraBytes = data.size() % intOrFp.getNearestByteSize();
  if (extraBytes)
    return emitError() << "provided raw data has " << extraBytes
                       << " extra bytes";
  return success();
}

/// Copy the data into a byte buffer aligned to the nearest power-of-2 byte
/// boundary.
static ArrayRef<uint8_t>
copyIntoAlignedBuffer(mlir::StorageUniquer::StorageAllocator &allocator,
                      ArrayRef<uint8_t> data, Type elementType) {
  unsigned byteSize =
      llvm::divideCeil(elementType.getIntOrFloatBitWidth(), CHAR_BIT);
  auto *ptr = static_cast<uint8_t *>(
      allocator.allocate(data.size(), llvm::NextPowerOf2(byteSize)));
  std::uninitialized_copy(data.begin(), data.end(), ptr);
  return {ptr, data.size()};
}

namespace {
/// Helper for parsing arbitrary integers and floats.
class PrimitiveElementParser {
public:
  PrimitiveElementParser(IntOrFPType type)
      : type(type), byteSize(type.getNearestByteSize()) {}

  /// Take the parsed data.
  std::vector<uint8_t> takeData() { return std::move(data); }

  /// Parse a single integer.
  ParseResult parseSingleInteger(AsmParser &p) {
    APInt apInt(type.getWidth(), 0, !type.isUnsignedInteger());
    if (p.parseInteger(apInt))
      return failure();
    append(apInt.sextOrTrunc(type.getWidth()));
    return success();
  }

  /// Parse a single float.
  ParseResult parseSingleFloat(AsmParser &p) {
    double fpVal;
    if (p.parseFloat(fpVal))
      return failure();
    APFloat apFloat(fpVal);
    // `double` is `f64`, so skip conversions when that's the case.
    if (!type.isF64()) {
      bool unused;
      apFloat.convert(type.cast<FloatType>().getFloatSemantics(),
                      APFloat::rmNearestTiesToEven, &unused);
    }
    append(apFloat.bitcastToAPInt());
    return success();
  }

  /// Parse a comma-separated list of integers or floats.
  ParseResult parseElements(AsmParser &p) {
    if (type.isa<FloatType>())
      return p.parseCommaSeparatedList([&] { return parseSingleFloat(p); });
    return p.parseCommaSeparatedList([&] { return parseSingleInteger(p); });
  }

private:
  /// Append one element.
  void append(const APInt &value) {
    size_t offset = data.size();
    data.insert(data.end(), byteSize, 0);
    llvm::StoreIntToMemory(value, data.data() + offset, byteSize);
  }

  IntOrFPType type;
  unsigned byteSize;
  std::vector<uint8_t> data;
};

/// Helper for printing arbitrary integers and floats.
class PrimitiveElementPrinter {
public:
  PrimitiveElementPrinter(IntOrFPType type, ArrayRef<uint8_t> data)
      : type(type), byteSize(type.getNearestByteSize()), data(data) {}

  /// Print a single integer.
  void printSingleInteger(AsmPrinter &p, unsigned i) {
    APInt intVal = getValue(i);
    // Print i1 as 'true' or 'false'.
    if (type.isInteger(1))
      p << (intVal.isOne() ? "true" : "false");
    else
      intVal.print(p.getStream(), !type.isUnsignedInteger());
  }

  /// Print a single float.
  void printSingleFloat(AsmPrinter &p, unsigned i) {
    APInt intVal = getValue(i);
    APFloat fpVal(type.cast<FloatType>().getFloatSemantics(), intVal);
    p.printFloat(fpVal);
  }

  /// Print the elements.
  void printElements(AsmPrinter &p) {
    unsigned size = data.size() / byteSize;
    if (auto fpType = dyn_cast<FloatType>(type)) {
      llvm::interleaveComma(llvm::seq<unsigned>(0, size), p,
                            [&](unsigned i) { printSingleFloat(p, i); });
    } else {
      llvm::interleaveComma(llvm::seq<unsigned>(0, size), p,
                            [&](unsigned i) { printSingleInteger(p, i); });
    }
  }

  /// Load a single element.
  APInt getValue(unsigned i) {
    APInt value(type.getWidth(), 0, !type.isUnsignedInteger());
    llvm::LoadIntFromMemory(value, data.data() + i * byteSize, byteSize);
    return value;
  }

private:
  IntOrFPType type;
  unsigned byteSize;
  ArrayRef<uint8_t> data;
};
} // namespace

/// Parse the elements of a primitive array.
static ParseResult parsePrimitiveArray(AsmParser &p,
                                       FailureOr<std::vector<uint8_t>> &values,
                                       Type elementType) {
  auto intOrFp = elementType.dyn_cast<IntOrFPType>();
  if (!intOrFp)
    return p.emitError(p.getCurrentLocation(),
                       "expected integer or float element type");

  // The array is empty if there are no colons.
  if (p.parseOptionalColon()) {
    values.emplace();
    return success();
  }

  PrimitiveElementParser handler(intOrFp);
  if (handler.parseElements(p))
    return failure();
  values = handler.takeData();
  return success();
}

/// Print the elements of a primitive array.
static void printPrimitiveArray(AsmPrinter &p, ArrayRef<uint8_t> values,
                                Type elementType) {
  // Skip the colon if the array is empty.
  if (values.empty())
    return;
  p << ": ";
  PrimitiveElementPrinter handler(elementType.cast<IntOrFPType>(), values);
  handler.printElements(p);
}

int64_t PrimitiveArrayAttr::size() const {
  return getData().size() /
         getElementType().cast<IntOrFPType>().getNearestByteSize();
}

PrimitiveArrayAttr PrimitiveArrayAttr::get(ArrayRef<uint8_t> data,
                                           Type elementType) {
  return get(elementType.getContext(), data, elementType);
}

//===----------------------------------------------------------------------===//
// ArrayElementsAttr
//===----------------------------------------------------------------------===//

/// Verify that the shaped type elements count matches the size of the array.
LogicalResult
ArrayElementsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          PrimitiveArrayAttr data, ShapedType type) {
  if (!type.hasStaticShape())
    return emitError() << "shaped type must have static shape";
  if (type.getNumElements() != data.size())
    return emitError() << "attribute type indicates " << type.getNumElements()
                       << " elements, but array has " << data.size();
  return success();
}

/// Parse the elements of an array elements attribute.
Attribute ArrayElementsAttr::parse(AsmParser &p, Type attrType) {
  // Validate the self type.
  auto type = dyn_cast_if_present<ShapedType>(attrType);
  if (!type) {
    p.emitError(p.getCurrentLocation(), "expected a shaped type");
    return {};
  }
  auto elementType = llvm::dyn_cast<IntOrFPType>(type.getElementType());
  if (!elementType) {
    p.emitError(p.getCurrentLocation(),
                "expected integer or float element type");
    return {};
  }

  auto emitError = [&] { return p.emitError(p.getCurrentLocation()); };

  if (p.parseLess())
    return {};
  // Check for an empty attribute.
  if (succeeded(p.parseOptionalGreater()))
    return getChecked(emitError, p.getContext(),
                      PrimitiveArrayAttr::get({}, elementType), type);

  FailureOr<std::vector<uint8_t>> result;
  PrimitiveElementParser handler(elementType);
  if (handler.parseElements(p) || p.parseGreater())
    return {};
  return getChecked(emitError, p.getContext(),
                    PrimitiveArrayAttr::get(handler.takeData(), elementType),
                    type);
}

/// Print the elements of an array elements attribute.
void ArrayElementsAttr::print(AsmPrinter &p) const {
  p << '<';
  PrimitiveElementPrinter handler(getElementType().cast<IntOrFPType>(),
                                  getData().getData());
  handler.printElements(p);
  p << '>';
}

ArrayElementsAttr ArrayElementsAttr::get(ArrayRef<uint8_t> data,
                                         ShapedType type) {
  return get(type.getContext(),
             PrimitiveArrayAttr::get(data, type.getElementType()), type);
}

ArrayRef<uint8_t> ArrayElementsAttr::getRawData() const {
  return getData().getData();
}

FailureOr<detail::AttrIterator>
ArrayElementsAttr::try_value_begin_impl(OverloadToken<Attribute>) const {
  return detail::AttrIterator(getRawData().data(), 0, getElementType());
}

Attribute detail::AttrIterator::operator*() const {
  auto type = elementType.cast<IntOrFPType>();
  APInt val(type.getWidth(), 0);
  unsigned byteSize = type.getNearestByteSize();
  llvm::LoadIntFromMemory(val, getBase() + getIndex() * byteSize, byteSize);
  if (type.isa<IntegerType>())
    return IntegerAttr::get(type, val);
  APFloat fpVal(type.cast<FloatType>().getFloatSemantics(), val);
  return FloatAttr::get(type.cast<FloatType>(), fpVal);
}

//===----------------------------------------------------------------------===//
// Shared Logic
//===----------------------------------------------------------------------===//

/// Pack the integer values into a byte array; The input template argument is
/// expected to be either an APInt or an APSInt.
template <typename Int>
static std::vector<uint8_t> packIntegerValues(unsigned width,
                                              ArrayRef<Int> values) {
  static_assert(std::is_same_v<Int, APInt> || std::is_same_v<Int, APSInt>,
                "unexpected integer type");
  unsigned byteSize = llvm::divideCeil(width, CHAR_BIT);
  std::vector<uint8_t> data(values.size() * byteSize, 0);
  for (auto &it : llvm::enumerate(values))
    llvm::StoreIntToMemory(it.value(), data.data() + (it.index() * byteSize),
                           byteSize);
  return data;
}

//===----------------------------------------------------------------------===//
// IntArrayElementsAttr
//===----------------------------------------------------------------------===//

IntArrayElementsAttr IntArrayElementsAttr::get(ShapedType type,
                                               ArrayRef<APInt> values) {
  std::vector<uint8_t> data =
      packIntegerValues(type.getElementTypeBitWidth(), values);
  return ArrayElementsAttr::get(data, type).cast<IntArrayElementsAttr>();
}

IntArrayElementsAttr IntArrayElementsAttr::get(ShapedType type,
                                               ArrayRef<APSInt> values) {
  std::vector<uint8_t> data =
      packIntegerValues(type.getElementTypeBitWidth(), values);
  return ArrayElementsAttr::get(data, type).cast<IntArrayElementsAttr>();
}

APInt IntArrayElementsAttr::Iterator::operator*() const {
  unsigned byteWidth = llvm::divideCeil(type.getWidth(), CHAR_BIT);
  APInt value(type.getWidth(), 0, !type.isUnsigned());
  llvm::LoadIntFromMemory(value, (const uint8_t *)base + index * byteWidth,
                          byteWidth);
  return value;
}

auto IntArrayElementsAttr::begin() const -> Iterator {
  return Iterator(getElementType().cast<IntegerType>(), getRawData().data(), 0);
}

auto IntArrayElementsAttr::end() const -> Iterator {
  return Iterator(getElementType().cast<IntegerType>(), getRawData().data(),
                  size());
}

bool IntArrayElementsAttr::classof(Attribute attr) {
  if (auto arr = llvm::dyn_cast<ArrayElementsAttr>(attr))
    return arr.getElementType().isa<IntegerType>();
  return false;
}

//===----------------------------------------------------------------------===//
// custom<DenseIntArray>
//===----------------------------------------------------------------------===//

ParseResult M::parseDenseIntArray(AsmParser &p, IntArrayElementsAttr &result,
                                  unsigned width,
                                  IntegerType::SignednessSemantics signedness) {
  auto elementType = IntegerType::get(p.getContext(), width, signedness);
  APInt value;
  mlir::OptionalParseResult maybeEmpty = p.parseOptionalInteger(value);
  // Check for an empty array.
  if (!maybeEmpty.has_value()) {
    result = IntArrayElementsAttr::get(ArrayType::get(0, elementType),
                                       ArrayRef<APInt>());
    return success();
  }
  if (maybeEmpty.value())
    return failure();

  SmallVector<APInt> values;
  auto addValue = [&](const APInt &value) {
    values.push_back(value.sextOrTrunc(elementType.getWidth()));
  };
  addValue(value);

  while (succeeded(p.parseOptionalComma())) {
    if (p.parseInteger(value))
      return failure();
    addValue(value);
  }
  result = IntArrayElementsAttr::get(ArrayType::get(values.size(), elementType),
                                     values);
  return success();
}

void M::printDenseIntArray(AsmPrinter &p, Operation *op,
                           IntArrayElementsAttr result, unsigned width,
                           IntegerType::SignednessSemantics) {
  llvm::interleaveComma(result, p);
}

//===----------------------------------------------------------------------===//
// FloatArrayElementsAttr
//===----------------------------------------------------------------------===//

FloatArrayElementsAttr FloatArrayElementsAttr::get(ShapedType type,
                                                   ArrayRef<APFloat> values) {
  SmallVector<APInt> intVals;
  intVals.reserve(values.size());
  for (const APFloat &value : values)
    intVals.push_back(value.bitcastToAPInt());
  std::vector<uint8_t> rawData = packIntegerValues(
      type.getElementTypeBitWidth(), llvm::makeArrayRef(intVals));
  return ArrayElementsAttr::get(rawData, type).cast<FloatArrayElementsAttr>();
}

APFloat FloatArrayElementsAttr::Iterator::operator*() const {
  FloatType type = this->type;
  unsigned byteWidth = llvm::divideCeil(type.getWidth(), CHAR_BIT);
  APInt intVal(type.getWidth(), 0);
  llvm::LoadIntFromMemory(intVal, (const uint8_t *)base + index * byteWidth,
                          byteWidth);
  return APFloat(type.getFloatSemantics(), intVal);
}

auto FloatArrayElementsAttr::begin() const -> Iterator {
  return Iterator(getElementType().cast<FloatType>(), getRawData().data(), 0);
}

auto FloatArrayElementsAttr::end() const -> Iterator {
  return Iterator(getElementType().cast<FloatType>(), getRawData().data(),
                  size());
}

bool FloatArrayElementsAttr::classof(Attribute attr) {
  if (auto arr = llvm::dyn_cast<ArrayElementsAttr>(attr))
    return arr.getElementType().isa<FloatType>();
  return false;
}

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a `DenseElementsAttr` to an `ArrayElementsAttr`. Pass through any
/// other kind of attribute. This should be the only place where the splatness
/// and bitpacked-ness of the attribute are handled.
Attribute M::convertDenseElements(Attribute attr) {
  auto denseElements = dyn_cast<DenseElementsAttr>(attr);
  if (!denseElements || !denseElements.getElementType().isIntOrFloat())
    return attr;
  if (denseElements.getType().getElementTypeBitWidth() % 8 == 0) {
    ArrayRef<char> charData = denseElements.getRawData();
    ArrayRef<uint8_t> data(reinterpret_cast<const uint8_t *>(charData.data()),
                           charData.size());
    // If the data is byte-aligned and is not splat, just pass it along.
    if (!denseElements.isSplat())
      return ArrayElementsAttr::get(data, denseElements.getType());

    // Replicate the splat.
    std::vector<uint8_t> replicated(data.size() * denseElements.size(), 0);
    for (unsigned i = 0, e = denseElements.size(); i < e; ++i)
      memcpy(replicated.data() + i * data.size(), data.data(), data.size());
    return ArrayElementsAttr::get(replicated, denseElements.getType());
  }

  // Unpack the data.
  if (denseElements.getElementType().isa<FloatType>()) {
    auto values = llvm::to_vector(denseElements.getValues<APFloat>());
    return FloatArrayElementsAttr::get(denseElements.getType(), values);
  }
  auto values = llvm::to_vector(denseElements.getValues<APInt>());
  return IntArrayElementsAttr::get(denseElements.getType(), values);
}

ElementsAttr
M::getAttrForTensorData(ShapedType type, StringRef bufferName,
                        ArrayRef<char> data,
                        DenseResourceElementsHandleManager &resourceManager) {
  size_t elementByteAlign =
      llvm::divideCeil(type.getElementTypeBitWidth(), CHAR_BIT);

  // When loading in a tensor, we make a distinction between the case where
  // the data is "small" and when it is "large". "large" data is stored as a
  // resource blob, while "small" data is stored inline in the context.
  if (!shouldUseOutOfLineAttrStorage(type.getNumElements()))
    return ArrayElementsAttr::get(
        {reinterpret_cast<const uint8_t *>(data.data()), data.size()}, type);

  // TODO: In many cases we should be able to use `UnmanagedAsmResourceBlob`
  // here and avoid all allocations. In these cases we need to ensure the
  // incoming buffer doesn't die before the generated MLIR.
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
      data, elementByteAlign);
  return DenseResourceElementsAttr::get(
      type, resourceManager.insert(bufferName, std::move(blob)));
}

//===----------------------------------------------------------------------===//
// AlignedBytesAttr
//===----------------------------------------------------------------------===//

/// Interns data into allocator with align (or 1 if no align constraint given).
/// The AlignedBytesAttr's getData will thus return a pointer respecting the
/// requested alignment.
static ArrayRef<uint8_t>
copyIntoBytes(mlir::StorageUniquer::StorageAllocator &allocator,
              ArrayRef<uint8_t> data, Optional<uint64_t> align) {
  auto *ptr = static_cast<uint8_t *>(
      allocator.allocate(data.size(), align ? *align : sizeof(uint8_t)));
  std::uninitialized_copy(data.begin(), data.end(), ptr);
  return {ptr, data.size()};
}

/// (Just to make clear we'll be converting from decoded hex strings directly
///  to vectors of uint8_t below.)
static_assert(sizeof(uint8_t) == sizeof(char));

/// Parses a hex string into data. The empty string denotes no data.
static ParseResult
parseAlignedBytesData(AsmParser &p, FailureOr<SmallVector<uint8_t>> &data) {
  std::string encoded;
  if (p.parseString(&encoded))
    return mlir::failure();
  if (encoded.empty()) {
    data.emplace();
    return success();
  }
  StringRef encodedRef = encoded;
  std::string decoded;
  auto startLoc = p.getCurrentLocation();
  if (!encodedRef.consume_front("0x") || encodedRef.empty() ||
      (encodedRef.size() & 1) || !llvm::tryGetFromHex(encodedRef, decoded)) {
    return p.emitError(startLoc, "invalid hex string for aligned_bytes");
  }
  data.emplace(decoded.size());
  memcpy(data->data(), decoded.data(), decoded.size());
  return success();
}

/// Print data as a hex string. The empty data array is printed as the empty
/// string.
static void printAlignedBytesData(AsmPrinter &p, ArrayRef<uint8_t> data) {
  if (data.empty()) {
    p << "\"\"";
    return;
  }
  p << "\"0x";
  StringRef decodedRef(reinterpret_cast<const char *>(data.data()),
                       data.size());
  p << llvm::toHex(decodedRef);
  p << "\"";
}

/// Verifies the attribute's align constraint is sensible.
LogicalResult
AlignedBytesAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         Optional<uint64_t> align,
                         ::llvm::ArrayRef<uint8_t> data) {
  if (align && !llvm::isPowerOf2_64(*align))
    return emitError() << "alignment must be a power of two.";
  return success();
}

/// ElementsAttrInterface: returns the underlying data.
FailureOr<const uint8_t *>
AlignedBytesAttr::try_value_begin_impl(OverloadToken<uint8_t>) const {
  return getData().data();
}

/// ElementsAttrInterface: returns the implied buffer type.
ShapedType AlignedBytesAttr::getType() const {
  return ArrayType::get(
      static_cast<int64_t>(getData().size()),
      IntegerType::get(getContext(), 8, IntegerType::Unsigned));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Support/MDialect/MAttrs.cpp.inc"
