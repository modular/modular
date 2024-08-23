//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ADT_REALLOCATINGTRIVIALVECTOR
#define SUPPORT_ADT_REALLOCATINGTRIVIALVECTOR

#include "llvm/Support/MathExtras.h"
#include <cstdlib>

namespace M {
/// This data structure is a resizable vector that contains only trivial types:
/// types which have trivial destructors, constructors, and copy constructors.
/// This vector is useful for carrying a varying number of such elements,
/// resizing with minimal memory pressure.
///
/// The vector doubles in capacity each time it increases in size.
template <typename T>
class ReallocatingTrivialVector {
public:
  static_assert(std::is_trivially_copy_constructible_v<T> &&
                    std::is_trivially_move_constructible_v<T> &&
                    std::is_trivially_destructible_v<T>,
                "T must be trivial");

  /// Initialize the vector with an initial size.
  explicit ReallocatingTrivialVector(unsigned initialSize)
      : ptr((T *)malloc(sizeof(T) * initialSize)), size(initialSize) {}

  ReallocatingTrivialVector() : ptr(nullptr), size(0) {}
  ReallocatingTrivialVector(ReallocatingTrivialVector<T> &&other)
      : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
    other.size = 0;
  }
  ~ReallocatingTrivialVector() { free(ptr); }

  /// Access the element at `i`.
  T &operator[](unsigned i) { return ptr[i]; }
  const T &operator[](unsigned i) const { return ptr[i]; }

  /// Ensure the vector can contain at least this many elements. This
  /// aggressively reserves up to the next power of 2 elements.
  void reserve(unsigned requiredSize) {
    requiredSize = llvm::NextPowerOf2(requiredSize);
    if (size >= requiredSize)
      return;
    // `realloc` copies the data over for us.
    ptr = (T *)realloc(ptr, sizeof(T) * requiredSize);
    size = requiredSize;
  }

  /// Directly access the underlying data.
  T *data() { return ptr; }
  const T *data() const { return ptr; }

private:
  T *ptr;
  unsigned size;
};
} // namespace M

#endif // SUPPORT_ADT_REALLOCATINGTRIVIALVECTOR
