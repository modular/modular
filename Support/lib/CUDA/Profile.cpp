//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CUDA/Profile.h"
#include "Utils.h"

using namespace M;
using namespace M::CUDA;

template <typename SymbolTy>
static ErrorOr<SymbolTy> fallibleGetCUDASymbol(std::string_view symbolName) {
  ErrorOr<llvm::sys::DynamicLibrary> cudaLib = getCUDADriverLibrary();
  if (cudaLib.isError())
    return cudaLib.takeError();

  return fallibleGetSymbol<SymbolTy>(*cudaLib, symbolName);
}

/// Enables profile collection by the active profiling tool for the current
/// context. If profiling is already enabled, then the function has no effect.
ErrorOrSuccess M::CUDA::profileStart() {
  static auto cuProfilerStart =
      fallibleGetCUDASymbol<void (*)()>("cuProfilerStart");
  if (cuProfilerStart.isError())
    return cuProfilerStart.takeError();

  (*cuProfilerStart)();
  return success();
}

/// Disables profile collection by the active profiling tool for the current
/// context. If profiling is already disabled, then the function has no effect.
ErrorOrSuccess M::CUDA::profileStop() {
  static auto cuProfilerEnd =
      fallibleGetCUDASymbol<void (*)()>("cuProfilerStop");
  if (cuProfilerEnd.isError())
    return cuProfilerEnd.takeError();

  (*cuProfilerEnd)();
  return success();
}
