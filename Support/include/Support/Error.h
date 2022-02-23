//===- Support/Error.h ------------------------------------------*- C++ -*-===//
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

#include "llvm/ADT/Twine.h"

namespace M {
template <typename T>
class ErrorOr;

/// This is a lightweight error class that holds a nul-terminated string, with a
/// static string optimization that does not allocate.  These are not implicitly
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
  /// Construct an ErrorOr with a static error string.
  template <size_t n>
  Error(const char (&message)[n]) : value(message), storageMode(kStaticError) {}

  /// Construct an ErrorOr with a dynamic Twine value (including std::string,
  /// const char *, etc).
  Error(llvm::Twine message) : storageMode(kMallocError) {
    llvm::SmallVector<char, 128> tmp;
    llvm::StringRef str = message.toStringRef(tmp);
    auto *ptr = (char *)malloc(str.size() + 1);
    memcpy(ptr, str.data(), str.size());
    ptr[str.size()] = 0;
    value = ptr;
  }

  Error(Error &&other) : value(other.value), storageMode(other.storageMode) {
    other.value = nullptr;
  }

  ~Error() {}

  /// Return the message this contains as a nul-terminated string.
  const char *get() const { return value; }

  /// Return the message this contains and release ownership of it.
  const char *release() {
    auto *result = value;
    storageMode = kStaticError;
    return result;
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
  Error(const Error &) = delete;
  Error &operator=(const Error &other) = delete;
  template <typename T>
  friend class M::ErrorOr;

  // Stored state.
  const char *value;
  StorageMode storageMode : 2;
};

} // end namespace M

#endif // SUPPORT_ERROR_H