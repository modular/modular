//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"

#include <mutex>

namespace M::AsyncRT {

// Forward declaration sufficient for pointer storage.
class Runtime;

namespace {

static std::mutex &getGlobalRuntimeMutexImpl() {
  static std::mutex m;
  return m;
}

static Runtime *&getGlobalRuntimePtrImpl() {
  static Runtime *ptr = nullptr;
  return ptr;
}

} // namespace

std::mutex &getGlobalRuntimeMutex() { return getGlobalRuntimeMutexImpl(); }

Runtime *getGlobalRuntimePointer() { return getGlobalRuntimePtrImpl(); }

void setGlobalRuntimePointer(Runtime *ptr) { getGlobalRuntimePtrImpl() = ptr; }

void clearGlobalRuntimePointerIfEquals(Runtime *ptr) {
  std::lock_guard<std::mutex> lock(getGlobalRuntimeMutexImpl());
  if (getGlobalRuntimePtrImpl() == ptr) {
    getGlobalRuntimePtrImpl() = nullptr;
  }
}

} // namespace M::AsyncRT
