//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STL_EXTRAS_H
#define SUPPORT_STL_EXTRAS_H

#include "Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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
  static ConditionallyOwnedPointer allocate(Args &&...args) {
    return ConditionallyOwnedPointer(new T(std::forward<Args>(args)...),
                                     /*shouldDelete=*/true);
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
// map_to_vector
//===----------------------------------------------------------------------===//

/// Map a range to a SmallVector with element types deduced from the mapping.
template <class ContainerTy, class FuncTy>
auto map_to_vector(ContainerTy &&C, FuncTy &&F) {
  return llvm::to_vector(
      llvm::map_range(std::forward<ContainerTy>(C), std::forward<FuncTy>(F)));
}

} // namespace M

#endif // SUPPORT_STL_EXTRAS_H
