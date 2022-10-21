//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_INVOKE_H
#define KGEN_SUPPORT_INVOKE_H

#include "Support/ML/DType.h"
#include "llvm/ADT/ArrayRef.h"

namespace M::KGEN {

namespace detail {

template <class T>
inline constexpr bool is_arrayref_v = false;

template <class T>
inline constexpr bool is_arrayref_v<ArrayRef<T>> = true;

template <class T>
inline constexpr bool is_arrayref_v<MutableArrayRef<T>> = true;

/// If the type is an ArrayRef, then destructure the ArrayRef into the form that
/// KGEN expects (a tuple of a pointer, size, and dtype).  For all other types,
/// just return the type as is.
template <typename T>
constexpr auto destructure_kgen_arguments(T &&arg) {
  if constexpr (is_arrayref_v<T>) {
    return std::tuple{(void *)arg.data(), (ssize_t)arg.size(),
                      DTypeForCXXType<typename T::value_type>::kind.getValue()};
  } else {
    return std::tuple{std::forward<T>(arg)};
  }
}

/// Destructure the arguments (returning a tuple).
template <typename First, typename... Rest>
constexpr auto destructure_kgen_arguments(First &&first, Rest &&...rest) {
  return std::tuple_cat(
      destructure_kgen_arguments(std::forward<First>(first)),
      destructure_kgen_arguments(std::forward<Rest>(rest)...));
}
} // namespace detail

/// Invoke a KGEN kernel with the given arguments. This would perform
/// destructuring so that ArrayRefs are passed as a tuple of (pointer, size,
/// dtype). All other types are passed as is.
template <typename F, typename... Args>
constexpr auto invoke(F &&f, Args &&...args) {
  return std::apply(std::forward<F>(f), detail::destructure_kgen_arguments(
                                            std::forward<Args>(args)...));
}

} // namespace M::KGEN

#endif // KGEN_SUPPORT_INVOKE_H
