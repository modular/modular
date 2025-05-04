//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

using namespace M;

/// Construct an Error with a dynamic Twine value (including std::string,
/// const char *, etc).
///
/// This is intentionally out of line, because we don't want error handling
/// logic bloating out libraries that produce the errors.
Error::Error(const llvm::Twine &message) : storageMode(kMallocError) {
  llvm::SmallVector<char, 128> tmp;
  llvm::StringRef str = message.toStringRef(tmp);
  assert(!str.empty() && "empty error strings are not allowed");
  auto *ptr = (char *)malloc(str.size() + 1);
  if (ptr == nullptr)
    std::abort();
  memcpy(ptr, str.data(), str.size());
  ptr[str.size()] = 0;
  value = ptr;
}

bool M::operator==(const Error &a, const Error &b) {
  return strcmp(a.get(), b.get()) == 0;
}

Error M::toModularError(llvm::Error error) {
  assert(error && "Successful (non-error) llvm::Error values do not have an "
                  "M::Error equivalent");
  return Error(llvm::toString(std::move(error)));
}
