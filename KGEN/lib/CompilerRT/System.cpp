//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <thread>

/// Returns the number of cores on the system.
extern "C" size_t KGEN_CompilerRT_CoreCount() {
  return std::thread::hardware_concurrency();
}
