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
  profiler.emplace(timeTraceGranularity, "kgen");

  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);

  outputFilePath = derived / "kgen.trace.json";
}

TraceProfiler::~TraceProfiler() {
  if (!profiler)
    return;
  if (auto err = profiler->write(outputFilePath.string(), "-"))
    llvm::errs() << "unable to write trace file: " << err.getError();
}
