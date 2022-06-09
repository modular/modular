//===- GenericML/Support/TensorShape.h ------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorShape class.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TENSORSHAPE_H
#define GENERICML_SUPPORT_TENSORSHAPE_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
namespace M {

namespace Detail {
/// This class implements a storage class to hold tensor shapes in a compact
/// 16-byte format that is suitable for long term storage on the heap.  It is
/// carefully laid out to hold common tensor sizes inline without losing support
/// for the full generality of tensor shapes.
class TensorShapeStorage {
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
  /// storage.  This is used to hold a TensorEltType for TensorSpec.  We keep
  /// this at the end of the storage so we can efficiently omit it from
  /// memset/memcpy operations.
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
  TensorShapeStorage() {
    representation.rep32.kind = RepKind::k32;
    representation.rep32.rank = 0;
    representation.rep32.auxillary = 0;
  }
  ~TensorShapeStorage() {
    if (isOutOfLine())
      delete[] representation.repOutOfLine.dims;
  }

  TensorShapeStorage(const TensorShapeStorage &other) {
    representation.rep32.kind = RepKind::k32;
    operator=(other);
  }
  TensorShapeStorage(TensorShapeStorage &&other) {
    representation.rep32.kind = RepKind::k32;
    operator=(other);
  }
  void operator=(const TensorShapeStorage &other) {
    memcpy(&representation, &other.representation, sizeof(representation));
    if (isOutOfLine()) {
      representation.repOutOfLine.dims = new ssize_t[getRank()];
      memcpy(representation.repOutOfLine.dims,
             other.representation.repOutOfLine.dims, getRank() * sizeof(ssize_t));
    }
  }
  void operator=(TensorShapeStorage &&other) {
    memcpy(&representation, &other.representation, sizeof(representation));
    // Take ownership of an out-of-line pointer if present.
    other.representation.repOutOfLine.kind = RepKind::k16;
  }

  /// Read out element.
  ssize_t operator[](size_t idx) const {
    assert(idx < getRank() && "invalid dimension #");
    auto rep = getRepKind();
    if (rep == RepKind::k32)
      return idx != 3 ? representation.rep32.dims[idx]
                      : representation.rep32.dim3;
    if (rep == RepKind::k16)
      return representation.rep16.dims[idx];
    return representation.repOutOfLine.dims[idx];
  }

  // Returns the rank.
  size_t getRank() const {
    static_assert(offsetof(Rep16, rank) == offsetof(Rep32, rank) &&
                      offsetof(Rep16, rank) == offsetof(RepOutOfLine, rank),
                  "Layout mismatch inside of TensorShape");
    // Because all of the representations store their rank in the same place, we
    // can just access an arbitrary one.
    return representation.rep16.rank;
  }

  // Provide access to the auxillary storage.
  uint8_t getAuxillary() const {
    static_assert(offsetof(Rep16, auxillary) == offsetof(Rep32, auxillary) &&
                      offsetof(Rep16, auxillary) ==
                          offsetof(RepOutOfLine, auxillary),
                  "Layout mismatch inside of TensorShape");
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

    iterator(const TensorShapeStorage *shape, size_t dimIdx)
        : shape(shape), dimIdx(dimIdx) {}

    iterator &operator+=(Base::difference_type n) {
      dimIdx += n;
      return *this;
    }
    iterator &operator-=(Base::difference_type n) {
      dimIdx -= n;
      return *this;
    }
    Base::difference_type operator-(iterator rhs) const {
      assert(shape == rhs.shape && "iterators from different shapes!");
      return Base::difference_type(dimIdx - rhs.dimIdx);
    }
    bool operator==(const iterator &rhs) const {
      assert(shape == rhs.shape && "iterators from different shapes!");
      return dimIdx == rhs.dimIdx;
    }
    ssize_t operator*() const { return (*shape)[dimIdx]; }

  private:
    const TensorShapeStorage *shape;
    size_t dimIdx;
  };

  // We cannot support mutation through the iterator.
  using const_iterator = iterator;
  iterator begin() const { return iterator(this, 0); }
  iterator end() const { return iterator(this, getRank()); }

  /// Bulk reassignment of elements.
  void assign(ArrayRef<ssize_t> elements);

  bool equalsIncludingAux(const TensorShapeStorage &rhs) const {
    if (isOutOfLine())
      return equalsIncludingAuxOOL(rhs);
    return memcmp(&representation, &rhs.representation,
                  sizeof(representation)) == 0;
  }

  bool equalsExcludingAux(const TensorShapeStorage &rhs) const {
    if (isOutOfLine())
      return equalsExcludingAuxOOL(rhs);
    // The aux field is the last byte of the representation.
    return memcmp(&representation, &rhs.representation,
                  sizeof(representation) - 1) == 0;
  }

private:
  bool equalsIncludingAuxOOL(const TensorShapeStorage &rhs) const;
  bool equalsExcludingAuxOOL(const TensorShapeStorage &rhs) const;

  // Return the storage representation for this TensorShape.
  RepKind getRepKind() const {
    // Check the representations line up.
    static_assert(offsetof(Rep16, kind) == offsetof(Rep32, kind) &&
                      offsetof(Rep16, kind) == offsetof(RepOutOfLine, kind),
                  "Layout mismatch inside of TensorShape");
    // Because all of the representations store their kind in the same place, we
    // can just access an arbitrary one.
    return representation.rep16.kind;
  }

  bool isOutOfLine() const { return getRepKind() == RepKind::kOutOfLine; }
};
} // namespace Detail

class TensorShape {
public:
  // This class has value semantics, implementing standard constructors,
  // assignment, copy construction etc.
  TensorShape() {}
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape &operator=(TensorShape &&) = default;

  // Allow constructing from both 32/64-bit and signed/unsigned integer
  // elements.  These are defined explicitly (instead of as a template) so
  // implicit conversions from things like SmallVector will work.
  /*implicit*/ TensorShape(ArrayRef<int32_t> e) { assign(e.begin(), e.end()); }
  /*implicit*/ TensorShape(ArrayRef<int64_t> e) { assign(e.begin(), e.end()); }
  /*implicit*/ TensorShape(ArrayRef<uint32_t> e) { assign(e.begin(), e.end()); }
  /*implicit*/ TensorShape(ArrayRef<uint64_t> e) { assign(e.begin(), e.end()); }
  /*implicit*/ TensorShape(ArrayRef<ssize_t> elts) { this->operator=(elts); }
  /*implicit*/ TensorShape(ArrayRef<size_t> elts) { this->operator=(elts); }

  // Allow converting from a range of integer type, with elements that can be
  // converted to ssize_t.
  template <typename IteratorType>
  TensorShape(IteratorType begin, IteratorType end) {
    assign(begin, end);
  }

  TensorShape &operator=(ArrayRef<ssize_t> elements) {
    storage.assign(elements);
    return *this;
  }
  TensorShape &operator=(ArrayRef<size_t> elements) {
    // Pointer cast to avoid copying the elements.
    ArrayRef<ssize_t> castedElts((const ssize_t *)elements.data(),
                                 elements.size());
    return operator=(castedElts);
  }

  template <typename IteratorType>
  void assign(IteratorType begin, IteratorType end) {
    operator=(SmallVector<ssize_t, 6>(begin, end));
  }

  uint8_t getAuxillaryStorage() const { return storage.getAuxillary(); }
  void setAuxillaryStorage(uint8_t value) { storage.setAuxillary(value); }

  /// Return the number of dimensions in this shape.
  size_t getRank() const { return storage.getRank(); }

  /// Return the total number of elements in this tensor, which is the product
  /// of all the dimension sizes.
  size_t getNumElements() const {
    size_t result = 1;
    for (auto dim : *this)
      result *= dim;
    return result;
  }

  // Support the typical iteration and subscripting operations.
  using iterator = typename Detail::TensorShapeStorage::iterator;
  iterator begin() { return storage.begin(); }
  iterator end() { return storage.end(); }
  using const_iterator = typename Detail::TensorShapeStorage::const_iterator;
  const_iterator begin() const { return storage.begin(); }
  const_iterator end() const { return storage.end(); }

  ssize_t operator[](size_t i) const { return storage[i]; }

  /// Return the dimensions as an unpacked SmallVector.
  SmallVector<ssize_t, 5> getDims() const {
    return SmallVector<ssize_t, 5>(begin(), end());
  }

  bool operator==(const TensorShape &rhs) const {
    return storage.equalsExcludingAux(rhs.storage);
  }
  bool operator!=(const TensorShape &rhs) const { return !(*this == rhs); }

  std::string getAsString() const;
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  Detail::TensorShapeStorage storage;
};

static_assert(sizeof(TensorShape) == 16, "TensorShape should not grow");

inline raw_ostream &operator<<(raw_ostream &os, const TensorShape &value) {
  value.print(os);
  return os;
}

} // end namespace M

#endif // GENERICML_SUPPORT_TENSORSHAPE_H
