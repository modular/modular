//===- Support/Error.h ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::Error type and related support logic.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ERROR_H
#define SUPPORT_ERROR_H

#include "Support/LLVMForwardDecls.h"

namespace M {
template <typename T>
class ErrorOr;

/// This is a lightweight error class that holds a nul-terminated string, with a
/// static string optimization that does not allocate.  This is not implicitly
/// copyable because that may invoke allocation, use the `copy()` method to make
/// this explicit if you want that.
///
/// By convention, error strings do not include a trailing \n in the string,
/// but do include a trailing period or other terminator.
///
class LLVM_NODISCARD Error final {
  enum StorageMode {
    kStaticError, // This contains a pointer to static data.  No allocation.
    kMallocError, // This contains a malloc'd string.
    kValue,       // This is a normal a value (used by ErrorOr).
  };

public:
  /// Construct an Error with a static error string.
  template <size_t n>
  Error(const char (&message)[n]) : value(message), storageMode(kStaticError) {}

  /// Construct an Error with a dynamic Twine value (including std::string,
  /// const char *, etc).
  Error(llvm::Twine message);

  /// Construct an Error with a known-static string that doesn't need lifetime
  /// management.
  static Error getStaticString(const char *message) {
    Error result;
    result.value = message;
    result.storageMode = kStaticError;
    return result;
  }

  Error(Error &&other) : value(other.value), storageMode(other.storageMode) {
    other.value = nullptr;
    other.storageMode = kStaticError;
  }

  ~Error() {
    if (storageMode == kMallocError)
      free(const_cast<void *>(static_cast<const void *>(value)));
  }

  /// Return the message this contains as a nul-terminated string.
  const char *get() const { return value; }

  /// Return the message this contains and release ownership of it.
  const char *release() {
    storageMode = kStaticError;
    return value;
  }

  // Explicit copy operation.
  Error copy() const {
    Error result;
    result.storageMode = storageMode;
    if (storageMode == kMallocError)
      result.value = strdup(value);
    else
      result.value = value;
    return result;
  }

  Error &operator=(Error &&other) {
    if (&other != this) {
      this->~Error();
      new (this) Error(std::move(other));
    }
    return *this;
  }

  /// Support [raw]ostream insertion.
  template <typename Stream>
  friend Stream &operator<<(Stream &os, const Error &error) {
    os << error.get();
    return os;
  }

private:
  Error() = default;
  Error(const Error &) = delete;                 // use copy() explicitly.
  Error &operator=(const Error &other) = delete; // use copy() explicitly.
  template <typename T>
  friend class M::ErrorOr;

  // Stored state.
  const char *value;
  StorageMode storageMode : 2;
};

} // end namespace M

#endif // SUPPORT_ERROR_H