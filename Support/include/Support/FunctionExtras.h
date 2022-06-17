//===- Support/FunctionExtras.h -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FUNCTION_EXTRAS_H
#define SUPPORT_FUNCTION_EXTRAS_H

#include <functional>
#include <type_traits>

namespace M {
/// This wraps a callable that may return a type or may return void, and allows
/// you to define custom return behavior in the case that the wrapped callable
/// returns void. The way you customize the return behavior is to provide a type
/// `Default` that has a static `get` method, and the result of that is what
/// will be returned. `Default` may be void, in which case this returns void.
template <typename Default, typename F, typename... Args,
          typename Result = std::invoke_result_t<F, Args...>>
static std::conditional_t<!std::is_void_v<Result>, Result, Default>
invokeWithDefaultResultType(F &&f, Args &&...args) {
  if constexpr (std::is_void_v<Result>) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    if constexpr (std::is_void_v<Default>)
      return;
    else
      return Default::get();
  } else {
    return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }
}
} // namespace M

#endif // SUPPORT_FUNCTION_EXTRAS_H
