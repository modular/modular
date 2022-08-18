//===- Support/STLExtras.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STL_EXTRAS_H
#define SUPPORT_STL_EXTRAS_H

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
} // namespace M

#endif // SUPPORT_STL_EXTRAS_H
