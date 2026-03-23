//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/RuntimeManager.h"
#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"

#include <optional>

namespace M::AsyncRT {

namespace {
/// Options used when the global runtime was first created. When a runtime
/// exists, getOrCreateRuntime must be called with the same options.
static std::optional<RuntimeOptions> &getStoredCreationOptions() {
  static std::optional<RuntimeOptions> opts;
  return opts;
}
} // namespace

RuntimeRef RuntimeManager::getOrCreateRuntime(RuntimeSource source,
                                              const RuntimeOptions &options) {
  std::lock_guard<std::mutex> lock(getGlobalRuntimeMutex());
  Runtime *ptr = getGlobalRuntimePointer();
  if (ptr) {
    std::optional<RuntimeOptions> &stored = getStoredCreationOptions();
    assert(stored && "creation options must be set when global runtime exists");
    assert(*stored == options &&
           "getOrCreateRuntime called with different options than used to "
           "create the global runtime");
    return RuntimeRef::copy(ptr);
  }
  getStoredCreationOptions().emplace(options);
  RuntimeRef ref = createRuntime(source, options);
  setGlobalRuntimePointer(ref.getPointer());
  return ref.copy();
}

} // namespace M::AsyncRT
