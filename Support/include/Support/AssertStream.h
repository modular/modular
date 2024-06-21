//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Assertions that can output values
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ASSERT_STREAM_H
#define SUPPORT_ASSERT_STREAM_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace M {

/// Structure holding information about a failed assertion. Allows user to
/// append information to the error message.
struct FailedAssertion {
  LLVM_ATTRIBUTE_NOINLINE FailedAssertion(std::string line, int64_t lineno) {
    getStorage().os << line << ":" << lineno << ": ";
  }
  [[noreturn]] ~FailedAssertion() {
    llvm::errs()
        << "=================================================================\n"
           "    !!! MAX AI Engine has encountered an internal error !!!\n"
           "=================================================================\n"
        << getStorage().os.str() << "\n";
    abort();
  }
  // Store error message info as a thread local struct. Avoids creating data for
  // each assertion which can blow the stack.
  struct Storage {
    std::string message;
    llvm::raw_string_ostream os;
    Storage() : message(), os(message) {}
  };
  LLVM_ATTRIBUTE_NOINLINE static Storage &getStorage() {
    static thread_local Storage storage;
    return storage;
  }
};
} // namespace M

/// Assert that a condition holds. Users can append addition information to the
/// error message by using the << operator.
/// ```
///   ASSERT_STREAM(false, << "This condition is always false!");
/// ```
#define ASSERT_STREAM(condition, message)                                      \
  if (LLVM_UNLIKELY(!(condition)))                                             \
  ::M::FailedAssertion(__FILE__, __LINE__).getStorage().os                     \
      << #condition << " is false.\n" message

#endif // SUPPORT_ASSERT_STREAM_H
