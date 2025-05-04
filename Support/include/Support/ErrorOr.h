//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ErrorOr and ErrorOrSuccess types, along with related
// support logic.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ERROR_OR_H
#define SUPPORT_ERROR_OR_H

#include "Support/Error.h"
#include "Support/LogicalResult.h"

namespace M {

// GCC's dataflow diagnostics get very confused and generate false
// positive warnings about freeing string literals, because it doesn't
// track the correlation with storageMode.  This is a dataflow diagnostic that
// gets triggered in the caller of ErrorOr code, not in the ErrorOr code, so
// there is nothing we can do locally to turn it off.  Turn it off globally.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif

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
class [[nodiscard]] ErrorOr {
  using StorageMode = Error::StorageMode;

public:
  using value_type =
      std::conditional_t<std::is_reference_v<T>,
                         std::reference_wrapper<std::remove_reference_t<T>>, T>;

private:
  using reference = std::remove_reference_t<T> &;
  using const_reference = const std::remove_reference_t<T> &;
  using pointer = std::remove_reference_t<T> *;
  using const_pointer = const std::remove_reference_t<T> *;

public:
  ~ErrorOr() {
    switch (storageMode) {
    case StorageMode::kValue:
      valueStorage.~value_type();
      return;
    case StorageMode::kStaticError:
      return;
    case StorageMode::kMallocError:
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
      std::free(const_cast<char *>(errorStorage));
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
      return;
    }
  }

  /// Construct from an error value produced by `error("string")`.
  ErrorOr(Error &&errorValue)
      : errorStorage(errorValue.value), storageMode(errorValue.storageMode) {
    errorValue.storageMode = Error::StorageMode::kStaticError;
  }

  /// Move constructor from value.
  template <class OtherT,
            typename = std::enable_if_t<std::is_convertible<OtherT, T>::value>>
  ErrorOr(OtherT &&val) : storageMode(StorageMode::kValue) {
    new (&valueStorage) value_type(std::forward<OtherT>(val));
  }

  /// Move constructor from ErrorOr.
  template <class OtherT,
            typename = std::enable_if_t<std::is_convertible<OtherT, T>::value>>
  ErrorOr(ErrorOr<OtherT> &&other) : storageMode(other.storageMode) {
    switch (storageMode) {
    case StorageMode::kValue:
      new (&valueStorage) value_type(std::forward<OtherT>(other.valueStorage));
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

  /// Convert this ErrorOr into a LogicalResult.
  /*implicit*/ operator LogicalResult() const {
    return failure(storageMode != Error::kValue);
  }

  /// Conversion to bool.  We allow conversion to bool, which allows testing
  /// and early exits with patterns like:
  ///
  ///    if (auto error = someThingThatMayFail())
  ///      return process(error);
  ///
  /// Compared to LogicalResult, there is lower chance of bool confusion here,
  /// because something will call takeError() to get the error out and that will
  /// crash if someone gets the logic wrong.
  explicit operator bool() const { return storageMode != Error::kValue; }

  /// Return true if this contains an error instead of a value.
  bool isError() const { return storageMode != Error::kValue; }

  reference get() {
    assert(storageMode == Error::kValue && "don't have a value!");
    return valueStorage;
  }

  const_reference get() const { return const_cast<ErrorOr<T> *>(this)->get(); }

  /// Given an ErrorOr with a value, take ownership of the underlying value away
  /// from the ErrorOr.
  value_type takeValue() { return std::move(get()); }

  const char *getError() const {
    assert(storageMode <= StorageMode::kValue && "invalid storage mode");
    switch (storageMode) {
    case StorageMode::kValue:
      return nullptr;
    case StorageMode::kStaticError:
    case StorageMode::kMallocError:
      return errorStorage;
    }
    llvm_unreachable("unsupported StorageMode");
  }

  /// Move the error out of this ErrorOr, taking ownership of any heap allocated
  /// data.
  Error takeError() {
    assert(storageMode <= StorageMode::kValue && "invalid storage mode");
    switch (storageMode) {
    case StorageMode::kValue:
      llvm::report_fatal_error("must hold an error");
    case StorageMode::kStaticError:
    case StorageMode::kMallocError: {
      Error result;
      result.storageMode = storageMode;
      result.value = errorStorage;
      storageMode = StorageMode::kStaticError;
      return result;
    }
    }
    llvm_unreachable("unsupported StorageMode");
  }

  pointer operator->() { return &get(); }
  reference operator*() { return get(); }
  const_pointer operator->() const { return &get(); }
  const_reference operator*() const { return get(); }

private:
  template <class OtherT>
  friend class ErrorOr;
  ErrorOr() = default;
  // Implicit copies are disabled, use copy() for explicit copies.
  ErrorOr(const ErrorOr &) = delete;                 // use copy() explicitly.
  ErrorOr &operator=(const ErrorOr &other) = delete; // use copy() explicitly.

  union {
    value_type valueStorage;
    const char *errorStorage;
  };
  StorageMode storageMode : 2;
};

template <typename T>
bool operator==(const ErrorOr<T> &a, const ErrorOr<T> &b) {
  if (a.isError() != b.isError())
    return false;
  if (a.isError())
    return strcmp(a.getError(), b.getError()) == 0;
  return a.get() == b.get();
}
template <typename T>
bool operator!=(const ErrorOr<T> &a, const ErrorOr<T> &b) {
  return !(a == b);
}

namespace Detail {
class Empty {};
} // namespace Detail

/// This type is used for APIs that either succeed (with no result value) or can
/// return an Error.
class [[nodiscard]] ErrorOrSuccess : public ErrorOr<Detail::Empty> {
public:
  using ErrorOr::ErrorOr;
  // This allows initialization from success().
  /*implicit*/ ErrorOrSuccess(SuccessType success) : ErrorOr(Detail::Empty()) {}

  // Allow default initialization to success.
  ErrorOrSuccess() : ErrorOr(Detail::Empty()) {}
};

/// Convert an LLVM error (which may be in either success state or error state)
/// to a Modular ErrorOrSuccess.
ErrorOrSuccess toModularErrorOr(llvm::Error llvmError);

/// Convert an LLVM Expected value to a Modular ErrorOr.
template <typename T>
ErrorOr<T> toModularErrorOr(llvm::Expected<T> expected) {
  if (expected)
    return std::move(*expected);
  return toModularError(expected.takeError());
}

/// Convert an LLVM ErrorOr value to a Modular ErrorOr.
template <typename T>
ErrorOr<T> toModularErrorOr(llvm::ErrorOr<T> expected) {
  if (expected)
    return std::move(*expected);
  return Error(expected.getError().message());
}

/// Error component enum and strings
#define ERROR_COMPONENT_EXPR                                                   \
  X(Unknown)                                                                   \
  X(ModularCLI)

enum class CodedErrorComponent {
#define X(val) val,
  ERROR_COMPONENT_EXPR
#undef X
};

#define X(val) static constexpr const char *CodedErrorComponent##val = #val;
ERROR_COMPONENT_EXPR
#undef X

/// CodedErrorOrSuccess enables structured telemetry or logging.
/// It is assumed the error id is a static string.
class CodedErrorOrSuccess {
public:
  CodedErrorOrSuccess(Error errorValue, CodedErrorComponent component,
                      const char *errorId)
      : isSuccess(false), error(std::move(errorValue)), component(component),
        errorId(errorId) {}

  CodedErrorOrSuccess(CodedErrorOrSuccess &&err)
      : isSuccess(err.isSuccess), error(std::move(err.error)),
        component(err.component), errorId(err.errorId) {}

  // This allows initialization from success().
  /*implicit*/ CodedErrorOrSuccess(SuccessType success) : isSuccess(true) {}

  bool isError() const { return !isSuccess; }

  /// Get the component in which the error occurred as a string
  const char *getComponentAsString() const;

  /// Get the specific error id as a string
  const char *getIdAsString() const { return errorId; }

  /// Get the internal Error as a string
  const char *getErrorAsString() const {
    assert(!isSuccess && "not an error!");
    return error->get();
  }

private:
  bool isSuccess;
  std::optional<Error> error;
  CodedErrorComponent component;
  const char *errorId;
};

} // namespace M

#endif // SUPPORT_ERROR_OR_H
