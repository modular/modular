//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RWMutex.h"
#include <type_traits>

#ifndef SUPPORT_THREADING_SHARED_H
#define SUPPORT_THREADING_SHARED_H

namespace M {

//===----------------------------------------------------------------------===//
// Shared
//===----------------------------------------------------------------------===//

/// This class guards a shared resource that can be accessed in read-only or
/// read-write mode. The resource can be a reference.
template <typename T>
class Shared {
  static constexpr bool is_ref = std::is_reference_v<T>;
  using reference = std::conditional_t<is_ref, T, T &>;
  using const_reference = std::conditional_t<is_ref, const T, const T &>;

public:
  Shared() : t() {}
  explicit Shared(T t) : t(std::forward<T>(t)) {}

  /// Get read-only access to the resource.
  template <typename FnT>
  auto read(FnT &&fn) {
    llvm::sys::SmartScopedReader<true> lock(mutex);
    return std::forward<FnT>(fn)(t);
  }

  /// Get modifiable access to the resource.
  template <typename FnT>
  auto modify(FnT &&fn) {
    llvm::sys::SmartScopedWriter<true> lock(mutex);
    return std::forward<FnT>(fn)(t);
  }

  /// Unsafe access to the underlying resource. This method does not lock the
  /// underlying resource and should only be used when it is known that the
  /// shared resource is being used by one thread.
  reference get() { return t; }

private:
  /// The shared resource.
  T t;
  /// The mutex guarding the shared resource.
  llvm::sys::SmartRWMutex<true> mutex;
};

} // namespace M

#endif // SUPPORT_THREADING_SHARED_H
