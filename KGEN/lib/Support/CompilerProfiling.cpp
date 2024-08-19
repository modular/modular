//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/CompilerProfiling.h"

using namespace M;
using namespace KGEN;

//===--------------------------------------------------------------------===//
// TraceProfiler
//===--------------------------------------------------------------------===//

void TraceProfiler::initialize(int timeTraceGranularity) {
  if constexpr (!KGEN::kIsTracingEnabled) {
    llvm::errs() << "[WARNING] -time-trace specified but tracing isn't coded "
                    "on; set `kIsTracingEnabled` to `true`";
  }

  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);

  profiler.emplace(timeTraceGranularity, "kgen",
                   (derived / "kgen.trace.json").string());
}

TraceProfiler::~TraceProfiler() {
  if (!profiler)
    return;
  if (auto err = profiler->write("-"))
    llvm::errs() << "unable to write trace file: " << err.getError();
}
