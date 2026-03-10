//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONTEXT_H
#define SUPPORT_CONTEXT_H

#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/FunctionExtras.h"
#include <memory>

namespace M {

class Context : public ReferenceCounted<Context> {
public:
  /// Transfers ptr into the context object set.
  template <typename T>
  void set(std::unique_ptr<T> ptr) {
    storage.set<T>(std::move(ptr));
  }

  /// Emplaces a new object of type T into the context object set and returns a
  /// reference to it.
  template <typename T, typename... Args>
  T &emplace(Args &&...args) {
    return storage.emplace<T, Args...>(std::forward<Args>(args)...);
  }

  /// Returns a reference to the object of type T held by the context object
  /// set. If it does not contain such an object, emplaces a new object and
  /// returns a reference to it.
  template <typename T, typename... Args>
  T &emplaceIfMissing(Args &&...args) {
    return storage.emplaceIfMissing<T, Args...>(std::forward<Args>(args)...);
  }

  /// Returns a pointer to the object of type T held by the context object set.
  /// If it does not contain such an object, calls the creator function to
  /// create one and install. Returns any error the creator function returns.
  template <typename T>
  ErrorOr<T *> createIfMissing(
      llvm::unique_function<ErrorOr<std::unique_ptr<T>>()> creator) {
    return storage.createIfMissing<T>(std::move(creator));
  }

  /// Returns a pointer to the context object of type T held by the context
  /// object set, or nullptr if no such object exists.
  template <typename T>
  T *get() {
    return storage.get<T>();
  }

  MODULAR_CXX_EXPORT ~Context();

private:
  GenericUniquePtrSet storage;
};

/// Convenience definitions.
using ContextRef = RCRef<Context>;

/// Global "current" Max context. Set by Init::createContext, cleared in
/// Context destructor. Visible to all threads. Asserts if nullptr (context
/// should always be set when in use). Returns a reference-counted ref.
MODULAR_CXX_EXPORT ContextRef getCurrentMaxContext();

/// Same as getCurrentMaxContext but returns nullptr if none set (no assert).
MODULAR_CXX_EXPORT Context *getCurrentMaxContextOrNull();

/// Sets the current Max context (visible to all threads). Stores
/// a raw pointer only; the global does not hold a ref. Cleared in ~Context()
/// when the last ContextRef to that context is released.
MODULAR_CXX_EXPORT void setCurrentMaxContext(Context *ptr);

} // namespace M

#endif // SUPPORT_CONTEXT_H
