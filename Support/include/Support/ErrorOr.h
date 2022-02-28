//===- Support/ErrorOr.h ----------------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::ErrorOr type and related support logic.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ERROR_OR_H
#define SUPPORT_ERROR_OR_H

#include "Support/Error.h"

namespace M {

/// ErrorOr<T> is a lightweight class that represents the result of an operation
/// or a string error.  This is designed to emulate the usage of returning a
/// pointer where nullptr indicates failure.  However instead of just knowing
/// that the operation failed, we also have an string error message that
/// describes why it failed.
///
/// It is used like the following:
/// \code
///   ErrorOr<Buffer> getBuffer() {
///     if (...not good...)
///       return error("buffer not found");
///     return myBuffer();
///   }
///
///   auto buffer = getBuffer();
///   if (const char *error = buffer.getError())
///     printf(error);
///   buffer->write("adena");
/// \endcode
///
///
/// Implicit conversion to bool returns true if there is a usable value. The
/// unary * and -> operators provide pointer like access to the value. Accessing
/// the value when there is an error will abort.
///
/// ErrorOr is moveable but not implicitly copyable because that may invoke
/// allocation of the value and error.  Typically you would want to move values
/// out of it.  However, you can use the `copy()` method to do explicit copies.
///
/// This class is extremely related to the llvm::ErrorOr<> type, except that it
/// holds a string error instead of an error code.  It is similar to the
/// llvm::Expected<> type but is much lighter weight in terms of code size and
/// header dependencies because it only holds character strings.
///
template <typename T>
class ErrorOr {
  using StorageMode = Error::StorageMode;

public:
  ~ErrorOr() {
    switch (storageMode) {
    case StorageMode::kValue:
      valueStorage.~T();
      return;
    case StorageMode::kStaticError:
      return;
    case StorageMode::kMallocError:
      std::free(const_cast<char *>(errorStorage));
      return;
    }
  }

  /// Construct from an error value produced by `error("string")`.
  ErrorOr(Error &&errorValue) : storageMode(errorValue.storageMode) {
    errorStorage = errorValue.release();
  }

  /// Move constructor from value.
  template <class OtherT,
            typename = std::enable_if_t<std::is_convertible<OtherT, T>::value>>
  ErrorOr(OtherT &&val) : storageMode(StorageMode::kValue) {
    new (&valueStorage) T(std::move(val));
  }

  /// Move constructor from ErrorOr.
  template <class OtherT,
            typename = std::enable_if_t<std::is_convertible<OtherT, T>::value>>
  ErrorOr(ErrorOr<OtherT> &&other) : storageMode(other.storageMode) {
    switch (storageMode) {
    case StorageMode::kValue:
      new (&valueStorage) T(std::move(other.valueStorage));
      return;
    case StorageMode::kStaticError:
      errorStorage = other.errorStorage;
      return;
    case StorageMode::kMallocError:
      errorStorage = other.errorStorage;
      other.errorStorage = nullptr;
      return;
    }
  }

  ErrorOr &operator=(ErrorOr &&other) {
    if (&other != this) {
      this->~ErrorOr();
      new (this) ErrorOr(std::move(other));
    }
    return *this;
  }

  ErrorOr copy() const {
    ErrorOr result;
    result.storageMode = storageMode;
    switch (storageMode) {
    case StorageMode::kValue:
      new (&result.valueStorage) T(valueStorage);
      break;
    case StorageMode::kStaticError:
      result.errorStorage = errorStorage;
      break;
    case StorageMode::kMallocError:
      result.errorStorage = strdup(errorStorage);
      break;
    }
    return result;
  }

  /// Return false if there is an error.
  explicit operator bool() const { return storageMode == Error::kValue; }

  T &get() {
    assert(storageMode == Error::kValue && "don't have a value!");
    return valueStorage;
  }
  const T &get() const { return const_cast<ErrorOr<T> *>(this)->get(); }

  const char *getError() const {
    switch (storageMode) {
    case StorageMode::kValue:
      return nullptr;
    case StorageMode::kStaticError:
    case StorageMode::kMallocError:
      return errorStorage;
    }
  }

  /// Move the error out of this ErrorOr, taking ownership of any heap allocated
  /// data.
  Error takeErrorStorage() {
    switch (storageMode) {
    case StorageMode::kValue:
      assert(0 && "must hold an error!");
    case StorageMode::kStaticError:
    case StorageMode::kMallocError:
      Error result;
      result.storageMode = storageMode;
      result.value = errorStorage;
      storageMode = StorageMode::kStaticError;
      return result;
    }
  }

  T *operator->() { return &get(); }
  T &operator*() { return get(); }
  const T *operator->() const { return &get(); }
  const T &operator*() const { return get(); }

private:
  template <class OtherT>
  friend class ErrorOr;
  ErrorOr(){};
  // Implicit copies are disabled, use copy() for explicit copies.
  ErrorOr(const ErrorOr &) = delete;                 // use copy() explicitly.
  ErrorOr &operator=(const ErrorOr &other) = delete; // use copy() explicitly.

  union {
    T valueStorage;
    const char *errorStorage;
  };
  StorageMode storageMode : 2;
};

} // namespace M

#endif // SUPPORT_ERROR_OR_H
