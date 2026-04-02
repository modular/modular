//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/RuntimeManager.h"
#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"

#include "llvm/Support/ErrorHandling.h"

#include <mutex>

namespace M::AsyncRT {

RuntimeRef getOrCreateRuntime(RuntimeSource source,
                              const RuntimeOptions &options) {
  std::lock_guard<std::mutex> lock(getGlobalRuntimeMutex());
  Runtime *existingRuntime = getGlobalRuntimePointer();
  if (existingRuntime) {
    if (getStoredGlobalRuntimeCreationOptions() != options)
      llvm::report_fatal_error(
          "AsyncRT::getOrCreateRuntime called requesting different options to "
          "those used to create the existing Runtime.");
    return RuntimeRef::copy(existingRuntime);
  }
  RuntimeRef newRuntime = createRuntime(source, options);
  getStoredGlobalRuntimeCreationOptions() = options;
  setGlobalRuntimePointer(newRuntime.getPointer());
  return newRuntime.copy();
}

} // namespace M::AsyncRT
