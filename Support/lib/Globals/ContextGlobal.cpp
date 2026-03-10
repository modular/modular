//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ContextGlobal.h"

#include <cassert>
#include <mutex>

namespace M {

/// Accessor for the global context mutex. Function-local static avoids
/// -Wglobal-constructors (and global destructor) on macOS/Clang.
static std::mutex &getGlobalContextMutex() {
  static std::mutex m;
  return m;
}

/// Accessor for the global context pointer. Kept in the same TU as the mutex.
static Context *&getGlobalContextPtr() {
  static Context *ptr = nullptr;
  return ptr;
}

Context *getCurrentMaxContextPointerOrNull() {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  return getGlobalContextPtr();
}

void setCurrentMaxContextPointer(Context *ptr) {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  getGlobalContextPtr() = ptr;
}

void clearGlobalContextPointerIfEquals(Context *ptr) {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  if (getGlobalContextPtr() == ptr) {
    getGlobalContextPtr() = nullptr;
  }
}

} // namespace M
