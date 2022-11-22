//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STL_EXTRAS_H
#define SUPPORT_STL_EXTRAS_H

#include "Support/LogicalResult.h"

namespace M {
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
  static ConditionallyOwnedPointer allocate(Args... args) {
    return ConditionallyOwnedPointer(std::forward<Args>(args)...);
  }

  /// Borrow the provided `T *`.
  static ConditionallyOwnedPointer borrow(T *ptr) {
    return ConditionallyOwnedPointer(ptr);
  }

  /// If `ptr` is provided, do not allocate a new pointer and borrow it.
  /// Otherwise, allocate a new one.
  template <typename... Args>
  static ConditionallyOwnedPointer allocateIfNeeded(T *ptr, Args... args) {
    if (ptr)
      return borrow(ptr);

    return allocate(std::forward<Args>(args)...);
  }

  /// A default instance of this class - it has nothing inside it.
  ConditionallyOwnedPointer() = default;

  /// Only delete the pointer if it's owned by this class.
  ~ConditionallyOwnedPointer() {
    if (shouldDelete)
      delete ptr;
  }

  /// Transparent accessors to get at the underlying pointer.
  T *operator->() { return ptr; }
  T &operator*() { return *ptr; }

private:
  ConditionallyOwnedPointer(T *ptr) : ptr(ptr), shouldDelete(false) {}
  template <typename... Args>
  ConditionallyOwnedPointer(Args... args)
      : ptr(new T(std::forward<Args>(args)...)), shouldDelete(true) {}

  T *ptr;
  bool shouldDelete;
};

} // namespace M

#endif // SUPPORT_STL_EXTRAS_H
