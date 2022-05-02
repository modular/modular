//===- MLSupport/TensorShape.h
//---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorShape, FixedRankTensorShape and
// CompactTensorShape classes.
//
// TODO: Implement FixedRankTensorShape/CompactTensorShape when needed.
//
//===----------------------------------------------------------------------===//

#ifndef COMMONML_TENSORSHAPE_H
#define COMMONML_TENSORSHAPE_H

#include "Support/ErrorOr.h"
#include "Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace M {

/// This is a common base class between TensorShape classes, providing a
/// consistent API.  This is parameterized on `DimensionStorageType` which holds
/// the dimension list itself.  This type is expected to have basic API for
/// iteration, size, and subscripting - modeled by things like SmallVector,
/// std::array, and custom storage types.
template <typename DimensionStorageType>
class TensorShapeImpl {
public:
  /// Return the number of dimensions in this shape.
  size_t getRank() const { return storage.size(); }

  /// Return the total number of elements in this tensor, which is the product
  /// of all the dimension sizes.
  size_t getNumElements() const {
    size_t result = 1;
    for (auto dim : *this)
      result *= dim;
    return result;
  }

  // Support the typical iteration and subscripting operations.
  using iterator = typename DimensionStorageType::iterator;
  iterator begin() { return storage.begin(); }
  iterator end() { return storage.end(); }
  using const_iterator = typename DimensionStorageType::const_iterator;
  const_iterator begin() const { return storage.begin(); }
  const_iterator end() const { return storage.end(); }

  const ssize_t &operator[](size_t i) const { return storage[i]; }
  ssize_t &operator[](size_t i) { return storage[i]; }

protected:
  TensorShapeImpl() {} // Derived class must choose to expose this (or not).
  DimensionStorageType storage;
};

/// This class represents a concrete (non-symbolic) shape of a Tensor value (for
/// use in a runtime) with standardized accessors and convenient methods
/// specific to shape handling.  This is a framework-independent class, which
/// doesn't have overly opinionated interpretation of its elements.  For
/// example, this class tolerates having the elements be negative, but doesn't
/// provide an interpretation for what that means.
///
/// NOTE: sizeof(TensorShape) is not small, so don't store large numbers of
/// these values in memory, use CompactTensorShape instead.  This is intended to
/// be used on the stack.
///
/// The storage for our dimensions has 5 inline elements to avoid allocations
/// in the common case.  "5" is a magic number, but it is precedented by both
/// PyTorch and TensorFlow Lite.
class TensorShape : public TensorShapeImpl<SmallVector<ssize_t, 5>> {
public:
  // This class has value semantics, implementing standard constructors,
  // assignment, copy construction etc.
  TensorShape() = default;
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape &operator=(const TensorShape &) = default;
  TensorShape &operator=(TensorShape &&) = default;

  // Allow constructing from both 32/64-bit and signed/unsigned integer
  // elements.  These are defined explicitly (instead of as a template) so
  // implicit conversions from things like SmallVector will work.
  /*implicit*/ TensorShape(ArrayRef<int32_t> elements) { assign(elements); }
  /*implicit*/ TensorShape(ArrayRef<int64_t> elements) { assign(elements); }
  /*implicit*/ TensorShape(ArrayRef<uint32_t> elements) { assign(elements); }
  /*implicit*/ TensorShape(ArrayRef<uint64_t> elements) { assign(elements); }

  template <typename EltType>
  void assign(ArrayRef<EltType> elements) {
    storage.assign(elements.begin(), elements.end());
  }

  void print(raw_ostream &os) const;
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &os, const TensorShape &value) {
  value.print(os);
  return os;
}

// TODO: Add CompactTensorShape when/if we care about dense packing of 16-byte
// values (e.g. when we have a native Tensor type).

// TODO: Add FixedRankTensorShape.

} // end namespace M

#endif // COMMONML_TENSORSHAPE_H
