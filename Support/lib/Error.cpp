//===- Error.cpp ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Error.h"
#include "llvm/ADT/Twine.h"
using namespace M;

/// Construct an ErrorOr with a dynamic Twine value (including std::string,
/// const char *, etc).
///
/// This is intentionally out of line, because we don't want error handling
/// logic bloating out libraries that that produce the errors.
Error::Error(llvm::Twine message) : storageMode(kMallocError) {
  llvm::SmallVector<char, 128> tmp;
  llvm::StringRef str = message.toStringRef(tmp);
  assert(!str.empty() && "empty error strings are not allowed");
  auto *ptr = (char *)malloc(str.size() + 1);
  memcpy(ptr, str.data(), str.size());
  ptr[str.size()] = 0;
  value = ptr;
}
