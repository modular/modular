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
#include "llvm/Support/raw_ostream.h"
namespace M {

template <size_t Rank>
class FixedRankTensorShape;

/// Print an array of dimensions as a shape.
void printShape(ArrayRef<ssize_t> dimensions, raw_ostream &os);

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
  TensorShape &operator=(TensorShape &&) = default;

  template <typename EltCollectionType>
  TensorShape &operator=(const EltCollectionType &elements) {
    storage.assign(elements.begin(), elements.end());
    return *this;
  }

  // Allow constructing from both 32/64-bit and signed/unsigned integer
  // elements.  These are defined explicitly (instead of as a template) so
  // implicit conversions from things like SmallVector will work.
  /*implicit*/ TensorShape(ArrayRef<int32_t> elements) { *this = elements; }
  /*implicit*/ TensorShape(ArrayRef<int64_t> elements) { *this = elements; }
  /*implicit*/ TensorShape(ArrayRef<uint32_t> elements) { *this = elements; }
  /*implicit*/ TensorShape(ArrayRef<uint64_t> elements) { *this = elements; }
  template <size_t Rank>
  /*implicit*/ TensorShape(const FixedRankTensorShape<Rank> &shape) {
    *this = shape;
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

/// This class models a TensorShape with a fixed rank (e.g. for kernels that are
/// working on 4D tensor values).  This allows them to be stored as those values
/// without other overhead.  This representation is not compressed, so it should
/// be used on the stack or in other places where size is not critical - not for
/// long term storage.
template <size_t Rank>
class FixedRankTensorShape : public TensorShapeImpl<std::array<ssize_t, Rank>> {
public:
  // This class has value semantics, implementing standard constructors,
  // assignment, copy construction etc.
  FixedRankTensorShape() { this->storage.fill(0); }
  FixedRankTensorShape(const FixedRankTensorShape &) = default;
  FixedRankTensorShape(FixedRankTensorShape &&) = default;
  FixedRankTensorShape &operator=(FixedRankTensorShape &&) = default;

  template <typename EltCollectionType>
  FixedRankTensorShape<Rank> &operator=(const EltCollectionType &elements) {
    assert(std::distance(elements.begin(), elements.end()) == Rank &&
           "incorrect rank for FixedRankTensorShape");
    std::copy(elements.begin(), elements.end(), this->storage.begin());
    return *this;
  }

  // Allow constructing from both 32/64-bit and signed/unsigned integer
  // elements.  These are defined explicitly (instead of as a template) so
  // implicit conversions from things like SmallVector will work.
  /*implicit*/ FixedRankTensorShape(ArrayRef<int32_t> elts) { *this = elts; }
  /*implicit*/ FixedRankTensorShape(ArrayRef<int64_t> elts) { *this = elts; }
  /*implicit*/ FixedRankTensorShape(ArrayRef<uint32_t> elts) { *this = elts; }
  /*implicit*/ FixedRankTensorShape(ArrayRef<uint64_t> elts) { *this = elts; }
  /*implicit*/ FixedRankTensorShape(const TensorShape &elts) { *this = elts; }

  void print(raw_ostream &os) const { printShape(this->storage, os); }
  void dump() const { print(llvm::errs()); }
};

} // end namespace M

#endif // COMMONML_TENSORSHAPE_H
