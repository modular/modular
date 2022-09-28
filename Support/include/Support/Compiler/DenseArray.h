//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_DENSEARRAY_H
#define SUPPORT_COMPILER_DENSEARRAY_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace M {

//===----------------------------------------------------------------------===//
// DenseIntArrayAttr
//===----------------------------------------------------------------------===//

/// This class represents a dense array of integers. Integers elements that do
/// not fit evenly into bytes are rounded up to the nearest byte.
class DenseIntArrayAttr : public DenseArrayAttr {
public:
  using DenseArrayAttr::DenseArrayAttr;

  /// Create an integer array. All APInt's must have the same width.
  static DenseIntArrayAttr get(IntegerType type, ArrayRef<APInt> values);

  /// Create an integer from an array of C++ values.
  template <typename IntT>
  static DenseIntArrayAttr get(MLIRContext *ctx, ArrayRef<IntT> values,
                               IntegerType::SignednessSemantics signedness) {
    auto type = IntegerType::get(ctx, sizeof(IntT) * CHAR_BIT, signedness);
    return DenseArrayAttr::get(RankedTensorType::get(values.size(), type),
                               {reinterpret_cast<const char *>(values.data()),
                                values.size() * sizeof(IntT)})
        .template cast<DenseIntArrayAttr>();
  }

  /// Iterate over the integer elements as APInt's.
  class Iterator
      : public llvm::indexed_accessor_iterator<Iterator, const char *, APInt,
                                               APInt, APInt> {
  public:
    APInt operator*() const;

  private:
    Iterator(IntegerType type, const char *base, size_t index)
        : indexed_accessor_iterator(base, index), type(type) {}

    friend class DenseIntArrayAttr;

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
    AsmParser &p, DenseIntArrayAttr &result, unsigned width,
    IntegerType::SignednessSemantics signedness = IntegerType::Signed);
void printDenseIntArray(
    AsmPrinter &p, Operation *op, DenseIntArrayAttr result, unsigned width,
    IntegerType::SignednessSemantics signedness = IntegerType::Signed);

//===----------------------------------------------------------------------===//
// DenseFloatArrayAttr
//===----------------------------------------------------------------------===//

/// This class represents a dense array of floats. Float elements that do not
/// fit evenly into bytes are rounded up to the nearest byte.
class DenseFloatArrayAttr : public DenseArrayAttr {
public:
  using DenseArrayAttr::DenseArrayAttr;

  /// Create a float array. All APFloat's must have the same width.
  static DenseFloatArrayAttr get(FloatType type, ArrayRef<APFloat> values);

  /// Iterate over the float elements as APFloat's.
  class Iterator
      : public llvm::indexed_accessor_iterator<Iterator, const char *, APFloat,
                                               APFloat, APFloat> {
  public:
    APFloat operator*() const;

  private:
    Iterator(FloatType type, const char *base, size_t index)
        : indexed_accessor_iterator(base, index), type(type) {}

    friend class DenseFloatArrayAttr;

    /// The element type.
    FloatType type;
  };

  Iterator begin() const;
  Iterator end() const;
  auto getValues() { return llvm::make_range(begin(), end()); }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};
} // namespace M

#endif // SUPPORT_COMPILER_DENSEARRAY_H
