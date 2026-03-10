//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ContextGlobal.h"

#include <cassert>
#include <mutex>

namespace M {

static std::mutex globalContextMutex;
static Context *globalContextPtr = nullptr;

Context *getCurrentMaxContextPointerOrNull() {
  std::lock_guard<std::mutex> lock(globalContextMutex);
  return globalContextPtr;
}

void setCurrentMaxContextPointer(Context *ptr) {
  std::lock_guard<std::mutex> lock(globalContextMutex);
  globalContextPtr = ptr;
}

void clearGlobalContextPointerIfEquals(Context *ptr) {
  std::lock_guard<std::mutex> lock(globalContextMutex);
  if (globalContextPtr == ptr) {
    globalContextPtr = nullptr;
  }
}

} // namespace M
