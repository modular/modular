//===- GenericML/Support/TensorShape.h ------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorShape, FixedRankTensorShape and
// CompactTensorShape classes.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TENSORSHAPE_H
#define GENERICML_SUPPORT_TENSORSHAPE_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
namespace M {

template <size_t Rank>
class FixedRankTensorShape;
class CompactTensorShape;

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
  /*implicit*/ TensorShape(const CompactTensorShape &elts) { *this = elts; }

  void print(raw_ostream &os) const;
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &os, const TensorShape &value) {
  value.print(os);
  return os;
}

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
  /*implicit*/ FixedRankTensorShape(const CompactTensorShape &elts) {
    *this = elts;
  }

  void print(raw_ostream &os) const { printShape(this->storage, os); }
  void dump() const { print(llvm::errs()); }
};

namespace Detail {
/// This class implements a storage class to hold tensor shapes in a compact
/// 16-byte format that is suitable for long term storage on the heap.  It is
/// carefully laid out to hold common tensor sizes inline without losing support
/// for the full generality of tensor shapes.
class CompactTensorShapeStorage {
  /// This supports two inline representations and an out-of-line one:
  ///  1) k16 can hold up to 6 dimensions when they fit into 16-bits.
  ///  2) k32 can hold up to 4 dimension where the first three fits in
  ///     32-bits and the last fits in 8 bits (typically channels or batch
  ///     size).
  ///  3) kOutOfLine is used for the general case.
  ///
  /// Important: Identical shapes have the same representation kind to allow
  /// efficient shape comparison with memcmp for k16 and k32.
  ///
  /// Each representation has an additional 8 bits of unused "auxillary"
  /// storage.  This is used to hold a TensorEltType for TensorSpec.
  enum class RepKind : uint8_t { k16, k32, kOutOfLine };

  struct Rep16 {
    int16_t dims[6];
    uint8_t unused;
    RepKind kind;
    uint8_t rank;
    uint8_t auxillary;
  };
  struct Rep32 {
    int32_t dims[3];
    int8_t dim3;
    RepKind kind;
    uint8_t rank;
    uint8_t auxillary;
  };

  struct RepOutOfLine {
    ssize_t *dims;
    // FIXME: This isn't correct for big endian systems, but we check with
    // static_assert below.
    uint8_t padding[13 - sizeof(void *)];
    RepKind kind;
    uint8_t rank;
    uint8_t auxillary;
  };

  union {
    Rep16 rep16;
    Rep32 rep32;
    RepOutOfLine repOutOfLine;
  } representation;

public:
  // Default construct to zero-D shape.
  CompactTensorShapeStorage() {
    representation.rep16.kind = RepKind::k16;
    representation.rep16.rank = 0;
    representation.rep16.auxillary = 0;
  }
  ~CompactTensorShapeStorage() {
    if (getRepKind() == RepKind::kOutOfLine)
      delete[] representation.repOutOfLine.dims;
  }

  CompactTensorShapeStorage(const CompactTensorShapeStorage &other) {
    representation.rep16.kind = RepKind::k16;
    operator=(other);
  }
  CompactTensorShapeStorage(CompactTensorShapeStorage &&other) {
    representation.rep16.kind = RepKind::k16;
    operator=(other);
  }
  void operator=(const CompactTensorShapeStorage &other) {
    memcpy(&representation, &other.representation, sizeof(representation));
    if (getRepKind() == RepKind::kOutOfLine) {
      representation.repOutOfLine.dims = new ssize_t[size()];
      memcpy(representation.repOutOfLine.dims,
             other.representation.repOutOfLine.dims, size() * sizeof(ssize_t));
    }
  }
  void operator=(CompactTensorShapeStorage &&other) {
    memcpy(&representation, &other.representation, sizeof(representation));
    // Take ownership of an out-of-line pointer if present.
    other.representation.repOutOfLine.kind = RepKind::k16;
  }

  /// Read out element.
  ssize_t operator[](size_t idx) const {
    auto rep = getRepKind();
    if (rep == RepKind::k32)
      return representation.rep32.dims[idx];
    if (rep == RepKind::k16)
      return representation.rep16.dims[idx];
    return representation.repOutOfLine.dims[idx];
  }

  // Returns the rank.
  size_t size() const {
    static_assert(offsetof(Rep16, rank) == offsetof(Rep32, rank) &&
                      offsetof(Rep16, rank) == offsetof(RepOutOfLine, rank),
                  "Layout mismatch inside of CompactTensorShape");
    // Because all of the representations store their rank in the same place, we
    // can just access an arbitrary one.
    return representation.rep16.rank;
  }

  // Provide access to the auxillary storage.
  uint8_t getAuxillary() const {
    static_assert(offsetof(Rep16, auxillary) == offsetof(Rep32, auxillary) &&
                      offsetof(Rep16, auxillary) ==
                          offsetof(RepOutOfLine, auxillary),
                  "Layout mismatch inside of CompactTensorShape");
    // Because all of the representations store their auxillary in the same
    // place, we can just access an arbitrary one.
    return representation.rep16.auxillary;
  }
  void setAuxillary(uint8_t value) { representation.rep16.auxillary = value; }

  /// Provides random access iteration, but only a read-only version.
  class iterator : public llvm::iterator_facade_base<
                       iterator, std::random_access_iterator_tag, ssize_t> {
  public:
    using Base =
        llvm::iterator_facade_base<iterator, std::random_access_iterator_tag,
                                   ssize_t>;

    iterator(const CompactTensorShapeStorage *shape, size_t dimIdx)
        : shape(shape), dimIdx(dimIdx) {}

    iterator &operator+=(Base::difference_type n) {
      dimIdx += n;
      return *this;
    }
    iterator &operator-=(Base::difference_type n) {
      dimIdx -= n;
      return *this;
    }
    Base::difference_type operator-(iterator rhs) {
      assert(shape == rhs.shape && "iterators from different shapes!");
      return Base::difference_type(dimIdx - rhs.dimIdx);
    }
    bool operator==(const iterator &rhs) const {
      assert(shape == rhs.shape && "iterators from different shapes!");
      return dimIdx == rhs.dimIdx;
    }
    ssize_t operator*() const { return (*shape)[dimIdx]; }

  private:
    const CompactTensorShapeStorage *shape;
    size_t dimIdx;
  };

  // We cannot support mutation through the iterator.
  using const_iterator = iterator;
  iterator begin() const { return iterator(this, 0); }
  iterator end() const { return iterator(this, size()); }

  // We do support bulk assignment.
  template <typename IteratorType>
  void assign(const IteratorType &beginIt, const IteratorType &endIt) {
    if (getRepKind() == RepKind::kOutOfLine)
      delete[] representation.repOutOfLine.dims;

    // Zero-initialize to ensure the representation value is determinsitic.
    memset(&representation, 0, sizeof(representation));

    // Get and set the rank, regardless of the representation.
    size_t rank = std::distance(beginIt, endIt);
    representation.repOutOfLine.rank = rank;
    assert(representation.repOutOfLine.rank == rank &&
           "can only handle rank up to 255");

    // Decide which representation we can use and initialize the elements.  The
    // most common case should fit into 4 dimensions.
    if (rank <= 4) {
      ssize_t dim;
      // Copy the iterator in case things don't work out.
      auto endItCopy = endIt;
      switch (rank) {
      default:
        assert(0 && "unreachable");
      case 4:
        dim = *--endItCopy;
        representation.rep32.dim3 = dim;
        if (representation.rep32.dim3 != dim)
          break; // Check for dimension too large.
        LLVM_FALLTHROUGH;
      case 3:
        dim = *--endItCopy;
        representation.rep32.dims[2] = dim;
        if (representation.rep32.dims[2] != dim)
          break; // Check for dimension too large.
        LLVM_FALLTHROUGH;
      case 2:
        dim = *--endItCopy;
        representation.rep32.dims[1] = dim;
        if (representation.rep32.dims[1] != dim)
          break; // Check for dimension too large.
        LLVM_FALLTHROUGH;
      case 1:
        dim = *--endItCopy;
        representation.rep32.dims[0] = dim;
        if (representation.rep32.dims[0] != dim)
          break; // Check for dimension too large.
        LLVM_FALLTHROUGH;
      case 0:
        representation.rep32.kind = RepKind::k32;
        return; // Success
      }
    }

    // Virtually everything else will fit into 7 dimensions.
    if (rank <= 7) {
      size_t i;
      // Copy the iterator in case things don't work out.
      auto beginItCopy = beginIt;
      for (i = 0; i < rank; ++i) {
        ssize_t dim = *beginItCopy;
        representation.rep16.dims[i] = dim;
        if (representation.rep16.dims[i] != dim)
          break;
      }
      if (i == rank) {
        representation.rep16.kind = RepKind::k16;
        return; // Success
      }
    }

    // Otherwise go out of line.
    representation.repOutOfLine.kind = RepKind::kOutOfLine;
    representation.repOutOfLine.dims = new ssize_t[rank];
    std::copy(beginIt, endIt, representation.repOutOfLine.dims);
  }

private:
  // Return the storage representation for this TensorShape.
  RepKind getRepKind() const {
    // Check the representations line up.
    static_assert(offsetof(Rep16, kind) == offsetof(Rep32, kind) &&
                      offsetof(Rep16, kind) == offsetof(RepOutOfLine, kind),
                  "Layout mismatch inside of CompactTensorShape");
    // Because all of the representations store their kind in the same place, we
    // can just access an arbitrary one.
    return representation.rep16.kind;
  }
};
} // namespace Detail

class CompactTensorShape
    : public TensorShapeImpl<Detail::CompactTensorShapeStorage> {
public:
  // This class has value semantics, implementing standard constructors,
  // assignment, copy construction etc.
  CompactTensorShape() {}
  CompactTensorShape(const CompactTensorShape &) = default;
  CompactTensorShape(CompactTensorShape &&) = default;
  CompactTensorShape &operator=(CompactTensorShape &&) = default;

  template <typename EltCollectionType>
  CompactTensorShape &operator=(const EltCollectionType &elements) {
    storage.assign(elements.begin(), elements.end());
    return *this;
  }

  // Allow constructing from both 32/64-bit and signed/unsigned integer
  // elements.  These are defined explicitly (instead of as a template) so
  // implicit conversions from things like SmallVector will work.
  /*implicit*/ CompactTensorShape(ArrayRef<int32_t> elts) { *this = elts; }
  /*implicit*/ CompactTensorShape(ArrayRef<int64_t> elts) { *this = elts; }
  /*implicit*/ CompactTensorShape(ArrayRef<uint32_t> elts) { *this = elts; }
  /*implicit*/ CompactTensorShape(ArrayRef<uint64_t> elts) { *this = elts; }
  /*implicit*/ CompactTensorShape(const TensorShape &elts) { *this = elts; }
  template <size_t Rank>
  /*implicit*/ CompactTensorShape(const FixedRankTensorShape<Rank> &shape) {
    *this = shape;
  }

  uint8_t getAuxillaryStorage() const { return storage.getAuxillary(); }
  void setAuxillaryStorage(uint8_t value) { storage.setAuxillary(value); }

  void print(raw_ostream &os) const;
  void dump() const;
};

static_assert(sizeof(CompactTensorShape) == 16, "TensorShape should not grow");

} // end namespace M

#endif // GENERICML_SUPPORT_TENSORSHAPE_H
