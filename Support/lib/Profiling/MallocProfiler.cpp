//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#if MODULAR_MALLOC_PROFILER

#include "Support/Profiling/MallocProfiler.h"
#include "llvm/ADT/bit.h"
#include <sstream>

#include "mimalloc.h"

std::string M::memoryStatistics() {
  std::ostringstream stats;

  mi_stats_print_out(
      [](const char *msg, void *stats) {
        *llvm::bit_cast<std::ostringstream *>(stats) << msg;
      },
      &stats);

  std::string result = stats.str();
  stats.clear();
  return result;
}

#endif // MODULAR_MALLOC_PROFILER
