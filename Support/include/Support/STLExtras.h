//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STL_EXTRAS_H
#define SUPPORT_STL_EXTRAS_H

#include "Support/AlignedAlloc.h"
#include "Support/LogicalResult.h"
#include <functional>
#include <type_traits>

namespace M {

/// Converts an enumeration to its underlying type. Note this function is
/// available as part of the STL in C++23.
template <typename Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

//===----------------------------------------------------------------------===//
// failableInterleave
//===----------------------------------------------------------------------===//

/// Call a function for each element in the range and a second function in
/// between every pair of elements. Either function can fail, in which case
/// iteration aborts and the function as a whole fails.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
auto failableInterleave(ForwardIterator begin, ForwardIterator end,
                        UnaryFunctor eachFn, NullaryFunctor betweenFn)
    -> decltype(betweenFn()) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    if (failed(betweenFn()) || failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
auto failableInterleave(const Container &c, UnaryFunctor eachFn,
                        NullaryFunctor betweenFn) {
  return failableInterleave(c.begin(), c.end(), eachFn, betweenFn);
}

//===----------------------------------------------------------------------===//
// ConditionallyOwnedPointer
//===----------------------------------------------------------------------===//

/// This class provides an ownership model for a pointer such that:
///  - If the class allocates the pointer, it'll delete it.
///  - If the class is passed-in the pointer, it will not delete it.
/// This serves the use case of optionally creating something in e.g. an MLIR
/// pass.
template <typename T>
struct ConditionallyOwnedPointer {
  /// Allocate a `T *` that this class will own (and therefore delete).
  template <typename... Args>
  static ConditionallyOwnedPointer allocate(Args &&...args) {
    return ConditionallyOwnedPointer(new T(std::forward<Args>(args)...),
                                     /*shouldDelete=*/true);
  }

  /// Allocate a `U *` that this class will own (and therefore delete). This
  /// overload allows the user to allocate a different type than `T` with the
  /// restriction that `U *` must be derived from `T *`.
  template <typename U, typename... Args>
  static ConditionallyOwnedPointer allocate(Args &&...args) {
    static_assert(std::is_base_of_v<T, U>, "`U` must be derived from `T`");
    return ConditionallyOwnedPointer(new U(std::forward<Args>(args)...),
                                     /*shouldDelete=*/true);
  }

  /// Take ownership of ptr. This overload allows the user to allocate a
  /// different type than `T` with the restriction that `U *` must be derived
  /// from `T *`.
  template <typename U>
  static ConditionallyOwnedPointer take(U *ptr) {
    static_assert(std::is_base_of_v<T, U>, "`U` must be derived from `T`");
    return ConditionallyOwnedPointer(ptr, /*shouldDelete=*/true);
  }

  /// Borrow the provided `T *`.
  static ConditionallyOwnedPointer borrow(T *ptr) {
    return ConditionallyOwnedPointer(ptr, /*shouldDelete=*/false);
  }

  /// Constructs 'null' pointer.
  ConditionallyOwnedPointer() = default;

  // No copying.
  ConditionallyOwnedPointer(const ConditionallyOwnedPointer &that) = delete;
  ConditionallyOwnedPointer &
  operator=(const ConditionallyOwnedPointer &that) = delete;

  // Can be moved.
  ConditionallyOwnedPointer(ConditionallyOwnedPointer &&that) { swap(that); }
  ConditionallyOwnedPointer &operator=(ConditionallyOwnedPointer &&that) {
    swap(that);
    return *this;
  }

  /// If `ptr` is provided, do not allocate a new pointer and borrow it.
  /// Otherwise, allocate a new one.
  template <typename... Args>
  static ConditionallyOwnedPointer allocateIfNeeded(T *ptr, Args &&...args) {
    if (ptr)
      return borrow(ptr);

    return allocate(std::forward<Args>(args)...);
  }

  /// If `ptr` is provided, do not allocate a new pointer and borrow it.
  /// Otherwise, take ownership of the result of calling createFn.
  static ConditionallyOwnedPointer takeIfNeeded(T *ptr,
                                                std::function<T *()> createFn) {
    if (ptr)
      return borrow(ptr);

    return take(createFn());
  }

  /// Only delete the pointer if it's owned by this class.
  ~ConditionallyOwnedPointer() {
    if (shouldDelete)
      delete ptr;
  }

  /// Transparent accessors to get at the underlying pointer.
  T *operator->() { return ptr; }
  const T *operator->() const { return ptr; }
  T &operator*() { return *ptr; }
  const T &operator*() const { return *ptr; }
  T *get() { return ptr; }
  const T *get() const { return ptr; }

  /// Check if this has a payload just like a normal pointer.
  explicit operator bool() const { return ptr != nullptr; }

private:
  ConditionallyOwnedPointer(T *ptr, bool shouldDelete)
      : ptr(ptr), shouldDelete(shouldDelete) {}

  void swap(ConditionallyOwnedPointer &that) {
    std::swap(ptr, that.ptr);
    std::swap(shouldDelete, that.shouldDelete);
  }

  T *ptr = nullptr;
  bool shouldDelete = false;
};

//===----------------------------------------------------------------------===//
// AlignedAllocator
//===----------------------------------------------------------------------===//

/// An allocator that can be used in STL data structures with a dynamic
/// alignment value.
template <typename T>
class AlignedAllocator {
public:
  using value_type = T;
  using pointer = T *;

  AlignedAllocator(size_t align) : align(align) {}

  pointer allocate(size_t n) { return (pointer)alignedAlloc(align, n); }

  void deallocate(pointer p, size_t n) { alignedFree(p); }

private:
  size_t align;
};

} // namespace M

#endif // SUPPORT_STL_EXTRAS_H
