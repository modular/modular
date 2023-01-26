//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MDIALECT_MATTRS_H
#define SUPPORT_MDIALECT_MATTRS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Triple.h"

//===----------------------------------------------------------------------===//
// ArrayElementsAttr
//===----------------------------------------------------------------------===//

namespace M::detail {
class AttrIterator
    : public llvm::indexed_accessor_iterator<AttrIterator, const uint8_t *,
                                             Attribute, Attribute, Attribute> {
public:
  AttrIterator(const uint8_t *data, size_t index, Type elementType)
      : indexed_accessor_iterator(data, index), elementType(elementType) {}

  Attribute operator*() const;

private:
  /// The element type.
  Type elementType;
};
} // namespace M::detail

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Support/MDialect/MAttrs.h.inc"

//===----------------------------------------------------------------------===//
// IntArrayElementsAttr
//===----------------------------------------------------------------------===//

namespace M {

/// This class represents a dense array of integers. Integer elements that do
/// not fit evenly into bytes are rounded up to the nearest byte.
class IntArrayElementsAttr : public ArrayElementsAttr {
public:
  using ArrayElementsAttr::ArrayElementsAttr;

  /// Create an integer array. All `APInt`s must have the same width.
  static IntArrayElementsAttr get(ShapedType type, ArrayRef<APInt> values);
  static IntArrayElementsAttr get(ShapedType type, ArrayRef<APSInt> values);

  /// Create an integer from an array of C++ values.
  template <typename IntT>
  static IntArrayElementsAttr get(MLIRContext *ctx, ArrayRef<IntT> values,
                                  IntegerType::SignednessSemantics signedness) {
    auto type = IntegerType::get(ctx, sizeof(IntT) * CHAR_BIT, signedness);
    return ArrayElementsAttr::get(
               {reinterpret_cast<const uint8_t *>(values.data()),
                values.size() * sizeof(IntT)},
               ArrayType::get(values.size(), type))
        .template cast<IntArrayElementsAttr>();
  }

  /// Iterate over the integer elements as `APInt`s.
  class Iterator
      : public llvm::indexed_accessor_iterator<Iterator, const uint8_t *, APInt,
                                               APInt, APInt> {
  public:
    APInt operator*() const;

  private:
    Iterator(IntegerType type, const uint8_t *base, size_t index)
        : indexed_accessor_iterator(base, index), type(type) {}

    friend class IntArrayElementsAttr;

    /// The element type.
    IntegerType type;
  };

  Iterator begin() const;
  Iterator end() const;
  auto getValues() { return llvm::make_range(begin(), end()); }

  template <typename IntT>
  ArrayRef<IntT> asArrayRef() {
    assert(sizeof(IntT) * CHAR_BIT == getElementType().getIntOrFloatBitWidth());
    return {reinterpret_cast<const IntT *>(getRawData().data()),
            static_cast<size_t>(size())};
  }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// custom<DenseIntArray>
//===----------------------------------------------------------------------===//

/// Parse or print an array of dense integers without the surrounding braces.
ParseResult parseDenseIntArray(
    AsmParser &p, IntArrayElementsAttr &result, unsigned width,
    IntegerType::SignednessSemantics signedness = IntegerType::Signed);
void printDenseIntArray(
    AsmPrinter &p, Operation *op, IntArrayElementsAttr result, unsigned width,
    IntegerType::SignednessSemantics signedness = IntegerType::Signed);

//===----------------------------------------------------------------------===//
// FloatArrayElementsAttr
//===----------------------------------------------------------------------===//

/// This class represents a dense array of floats. Float elements that do not
/// fit evenly into bytes are rounded up to the nearest byte.
class FloatArrayElementsAttr : public ArrayElementsAttr {
public:
  using ArrayElementsAttr::ArrayElementsAttr;

  /// Create a float array. All `APFloat`s must have the same width.
  static FloatArrayElementsAttr get(ShapedType type, ArrayRef<APFloat> values);

  /// Iterate over the float elements as `APFloat`s.
  class Iterator
      : public llvm::indexed_accessor_iterator<Iterator, const uint8_t *,
                                               APFloat, APFloat, APFloat> {
  public:
    APFloat operator*() const;

  private:
    Iterator(FloatType type, const uint8_t *base, size_t index)
        : indexed_accessor_iterator(base, index), type(type) {}

    friend class FloatArrayElementsAttr;

    /// The element type.
    FloatType type;
  };

  Iterator begin() const;
  Iterator end() const;
  auto getValues() { return llvm::make_range(begin(), end()); }

  template <typename FloatT>
  ArrayRef<FloatT> asArrayRef() {
    assert(sizeof(FloatT) * CHAR_BIT ==
           getElementType().getIntOrFloatBitWidth());
    return {reinterpret_cast<const FloatT *>(getRawData().data()),
            static_cast<size_t>(size())};
  }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// IndexArrayElementsAttr
//===----------------------------------------------------------------------===//

/// This class represents a dense array of indices. Index type elements are
/// stored according to the index type's internal storage bitwidth.
class IndexArrayElementsAttr : public ArrayElementsAttr {
public:
  using ArrayElementsAttr::ArrayElementsAttr;

  /// Create an index array.
  static IndexArrayElementsAttr get(ShapedType type, ArrayRef<int64_t> values);

  using iterator = ArrayRef<int64_t>::iterator;

  iterator begin() const { return asArrayRef().begin(); }
  iterator end() const { return asArrayRef().end(); }

  ArrayRef<int64_t> asArrayRef() const {
    return {reinterpret_cast<const int64_t *>(getRawData().data()),
            static_cast<size_t>(size())};
  }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a `DenseElementsAttr` to an `ArrayElementsAttr`. Pass through any
/// other kind of attribute. This should be the only place where the splatness
/// and bitpacked-ness of the attribute are handled.
Attribute convertDenseElements(Attribute attr);

/// Returns an attribute to store the given tensor data. Using forceOutOfLine
/// will force the binary to be stored as a dialect resource. Depending on the
/// amount of data and optional alignment, this might be inlined as:
///  - 'ArrayElementAttr' (small data, no alignment constraint)
///  - 'AlignedBytes' (small data, alignment constraint)
///  - 'DenseResourceElementsAttr' (large data, if no alignment constraint then
///    use the element type's bit width rounded up to whole bytes.
ElementsAttr
getAttrForTensorData(ShapedType type, StringRef bufferName, ArrayRef<char> data,
                     DenseResourceElementsHandleManager &resourceManager,
                     Optional<size_t> optAlignment = {},
                     bool forceOutOfLine = false);

} // namespace M

#endif // SUPPORT_MDIALECT_MATTRS_H
