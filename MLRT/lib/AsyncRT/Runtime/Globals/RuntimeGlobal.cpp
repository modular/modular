//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"

#include <mutex>

namespace M::MLRT {

namespace {

static std::mutex &getGlobalRuntimeMutexImpl() {
  static std::mutex m;
  return m;
}

static Runtime *&getGlobalRuntimePtrImpl() {
  static Runtime *ptr = nullptr;
  return ptr;
}

static RuntimeOptions &storedGlobalRuntimeCreationOptionsImpl() {
  static RuntimeOptions opts;
  return opts;
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

RuntimeOptions &getStoredGlobalRuntimeCreationOptions() {
  return storedGlobalRuntimeCreationOptionsImpl();
}

} // namespace M::MLRT
