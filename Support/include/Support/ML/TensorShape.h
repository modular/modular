//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorShape class.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_TENSORSHAPE_H
#define SUPPORT_ML_TENSORSHAPE_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace M {

/// The maximum tensor rank for any tensor shape.
/// This value must match max_rank in Kernels/mojo/Stdlib/Buffer.mojo
constexpr size_t kMaxRank = 8;

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
  /// Each representation has an additional 8 bits of unused "auxiliary"
  /// storage.  This is used to hold a DType for TensorSpec.  We keep
  /// this at the end of the storage so we can efficiently omit it from
  /// memset/memcpy operations.
  enum class RepKind : uint8_t { k16, k32, kOutOfLine };

  struct Rep16 {
    int16_t dims[6];
    uint8_t unused;
    RepKind kind;
    uint8_t rank;
    uint8_t auxiliary;
  };
  struct Rep32 {
    int32_t dims[3];
    int8_t dim3;
    RepKind kind;
    uint8_t rank;
    uint8_t auxiliary;
  };

  struct RepOutOfLine {
    ssize_t *dims;
    // FIXME: This isn't correct for big endian systems, but we check with
    // static_assert below.
    uint8_t padding[13 - sizeof(void *)];
    RepKind kind;
    uint8_t rank;
    uint8_t auxiliary;
  };

  union {
    Rep16 rep16;
    Rep32 rep32;
    RepOutOfLine repOutOfLine;
  } representation;

public:
  // Default construct to zero-D shape.
  TensorShapeStorage() {
    memset(&representation, 0, sizeof(representation));
    representation.repOutOfLine.kind = RepKind::k32;
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
             other.representation.repOutOfLine.dims,
             getRank() * sizeof(ssize_t));
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
    if (rep == RepKind::k32) {
      assert(idx <= 3 && "you can only fit 4 dimensions in k32");
      return idx != 3 ? representation.rep32.dims[idx]
                      : representation.rep32.dim3;
    }
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

  // Provide access to the auxiliary storage.
  uint8_t getAuxiliary() const {
    static_assert(offsetof(Rep16, auxiliary) == offsetof(Rep32, auxiliary) &&
                      offsetof(Rep16, auxiliary) ==
                          offsetof(RepOutOfLine, auxiliary),
                  "Layout mismatch inside of TensorShape");
    // Because all of the representations store their auxiliary in the same
    // place, we can just access an arbitrary one.
    return representation.rep16.auxiliary;
  }
  void setAuxiliary(uint8_t value) { representation.rep16.auxiliary = value; }

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
  /// TODO: Forcing dimensions to 64-bit is suboptimal on 32-bit hosts.
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

  RepKind getKind() const { return representation.rep16.kind; }

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

// NOTE: These assignment helpers could live in TensorShapeStorage or
// TensorShape however older gcc versions complain.
template <typename IteratorType>
void assign(TensorShapeStorage &storage, IteratorType begin, IteratorType end) {
  storage.assign(SmallVector<ssize_t, kMaxRank>(begin, end));
}

template <typename ElementType>
void assign(TensorShapeStorage &storage, ArrayRef<ElementType> elts) {
  assign(storage, elts.begin(), elts.end());
}

template <>
inline void assign<ssize_t>(TensorShapeStorage &storage,
                            ArrayRef<ssize_t> elts) {
  storage.assign(elts);
}

template <>
inline void assign<size_t>(TensorShapeStorage &storage, ArrayRef<size_t> elts) {
  // Pointer cast to avoid copying the elements.
  ArrayRef<ssize_t> castedElts((const ssize_t *)elts.data(), elts.size());
  storage.assign(castedElts);
}

} // namespace Detail

class TensorShape {
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
  /*implicit*/ TensorShape(ArrayRef<int32_t> elts) { assign(storage, elts); }
  /*implicit*/ TensorShape(ArrayRef<uint32_t> elts) { assign(storage, elts); }
  /*implicit*/ TensorShape(ArrayRef<int64_t> elts) { assign(storage, elts); }
  /*implicit*/ TensorShape(ArrayRef<uint64_t> elts) { assign(storage, elts); }
#ifdef __APPLE__
  /*implicit*/ TensorShape(ArrayRef<size_t> elts) { assign(storage, elts); }
  /*implicit*/ TensorShape(ArrayRef<ssize_t> elts) { assign(storage, elts); }
#endif // __APPLE__

  template <typename ElementType,
            typename = std::enable_if_t<std::is_integral_v<ElementType>>>
  TensorShape(const std::initializer_list<ElementType> &elts) {
    assign(storage, elts.begin(), elts.end());
  }

  // Allow converting from a range of integer type, with elements that can be
  // converted to ssize_t.
  template <typename IteratorType>
  TensorShape(IteratorType begin, IteratorType end) {
    assign(storage, begin, end);
  }

  uint8_t getAuxiliaryStorage() const { return storage.getAuxiliary(); }
  void setAuxiliaryStorage(uint8_t value) { storage.setAuxiliary(value); }

  /// Return the number of dimensions in this shape.
  size_t getRank() const { return storage.getRank(); }

  /// Return the underlying kind of the spec.
  uint8_t getKind() const { return (uint8_t)storage.getKind(); }

  /// Return the total number of elements in this tensor, which is the product
  /// of all the dimension sizes.
  size_t getNumElements() const {
    size_t result = 1;
    for (auto dim : *this) {
      assert(dim >= 0 && "attempting to get the number of elements of a "
                         "TensorSpec with unknown dimensions");
      result *= static_cast<size_t>(dim);
    }
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

  /// Return the dimensions as a temporary, unpacked SmallVector.
  ///
  /// CAUTION: You probably don't need to call this! TensorShape supports
  /// indexing, iteration and getRank directly.
  SmallVector<ssize_t, kMaxRank> getDimsCopy() const {
    return SmallVector<ssize_t, kMaxRank>(begin(), end());
  }

  bool operator==(const TensorShape &rhs) const {
    return storage.equalsExcludingAux(rhs.storage);
  }
  bool operator!=(const TensorShape &rhs) const { return !(*this == rhs); }

  std::string getAsString() const;
  void print(raw_ostream &os) const;
  void dump() const;

  /// Parses a string of the form dim0xdim1x...xdimN into a TensorShape.
  static ErrorOr<TensorShape> parseFromString(StringRef);

protected:
  Detail::TensorShapeStorage storage;
};

static_assert(sizeof(TensorShape) == 16, "TensorShape should not grow");

inline raw_ostream &operator<<(raw_ostream &os, const TensorShape &value) {
  value.print(os);
  return os;
}

} // namespace M

namespace llvm::yaml {

enum class QuotingType;

template <typename T, typename Enable>
struct ScalarTraits;

// Equivalent to LLVM_YAML_DECLARE_SCALAR_TRAITS, but without requiring
// including YAMLTraits.h (YAMLTraits.h is a large header and TensorShape.h is
// somewhat pervasive, so we don't want to make it too heavy to compile)
template <>
struct ScalarTraits<M::TensorShape, void> {
  static void output(const M::TensorShape &value, void *ctxt, raw_ostream &out);
  static StringRef input(StringRef scalar, void *ctxt, M::TensorShape &value);
  static QuotingType mustQuote(StringRef);
};

} // namespace llvm::yaml

#endif // SUPPORT_ML_TENSORSHAPE_H
