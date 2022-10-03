//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

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
  auto intOrFp = elementType.dyn_cast<IntOrFPType>();
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
    if (auto fpType = type.dyn_cast<FloatType>()) {
      llvm::interleaveComma(llvm::seq<unsigned>(0, size), p,
                            [&](unsigned i) { printSingleFloat(p, i); });
    } else {
      llvm::interleaveComma(llvm::seq<unsigned>(0, size), p,
                            [&](unsigned i) { printSingleInteger(p, i); });
    }
  }

private:
  /// Load a single element.
  APInt getValue(unsigned i) {
    APInt value(type.getWidth(), 0, !type.isUnsignedInteger());
    llvm::LoadIntFromMemory(value, data.data() + i * byteSize, byteSize);
    return value;
  }

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
  auto type = attrType.dyn_cast_or_null<ShapedType>();
  if (!type) {
    p.emitError(p.getCurrentLocation(), "expected a shaped type");
    return {};
  }
  auto elementType = type.getElementType().dyn_cast<IntOrFPType>();
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

ArrayElementsAttr ArrayElementsAttr::get(PrimitiveArrayAttr data,
                                         ShapedType type) {
  return get(data.getContext(), data, type);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Support/MDialect/MAttrs.cpp.inc"
