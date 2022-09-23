//===- Support/STLExtras.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STL_EXTRAS_H
#define SUPPORT_STL_EXTRAS_H

#include "Support/LogicalResult.h"
#include <type_traits>

namespace M {
#if defined(__cpp_lib_type_identity)
/// We are compiling with a compiler which knows about __cpp_lib_type_identity,
/// so we just use it.
template <class T>
using type_identity = std::type_identity<T>;
template <class T>
using type_identity_t = std::type_identity_t<T>;
#else  // defined(__cpp_lib_type_identity)
/// Otherwise, we define the struct that is equivalent of C++20
/// std::type_identity
///
/// TODO: This is dead code when we switch to C++20
template <class T>
struct type_identity {
  using type = T;
};
template <class T>
using type_identity_t = typename type_identity<T>::type;
#endif // defined(__cpp_lib_type_identity)

//===----------------------------------------------------------------------===//
// failableInterleave
//===----------------------------------------------------------------------===//

/// Call a function for each element in the range and a second function in
/// between every pair of elements. Either function can fail, in which case
/// iteration aborts and the function as a whole fails.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline auto failableInterleave(ForwardIterator begin, ForwardIterator end,
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
inline auto failableInterleave(const Container &c, UnaryFunctor eachFn,
                               NullaryFunctor betweenFn) {
  return failableInterleave(c.begin(), c.end(), eachFn, betweenFn);
}

} // namespace M

#endif // SUPPORT_STL_EXTRAS_H
