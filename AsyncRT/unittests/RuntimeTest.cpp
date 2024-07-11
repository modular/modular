//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::LLCL;

namespace {

std::unique_ptr<Runtime> createRuntime() {
  return LLCL::createUniqueRuntime(LLCL::RuntimeOptions()
                                       .withLeakCheckedAllocator()
                                       .withMainWillNotDonate());
}

/// Test to ensure that we can utilize the full range of indices for runtime.
/// This is mostly meant to be a precursor to check that the full range of
/// runtime indices is available
TEST(RuntimeTest, MaxRuntime) {
  std::vector<std::unique_ptr<Runtime>> allRuntimes;
  for (int i = 0; i < 255; ++i)
    allRuntimes.emplace_back(createRuntime());
  for (auto &runtime : allRuntimes)
    runtime.reset();
}

/// Test to ensure that we can utilize the full range of indices for runtime. It
/// checks the free indices first, and then fills up the index space. Next it
/// then removes 10 instances from the middle of the range and then attempts to
/// add 10 instances again which should succeed.
TEST(RuntimeTest, MaxRuntimeUtilize) {
  const uint8_t numRuntimes =
      M::LLCL::Detail::RuntimeTable::getSingleton().numActiveRuntimes();

  std::vector<std::unique_ptr<Runtime>> allRuntimes;
  for (uint8_t i = 0; i < (255 - numRuntimes); ++i) {
    allRuntimes.emplace_back(createRuntime());
  }
  // now remove 10 indices from the middle of the indices range
  for (uint8_t i = 0; i < 10; ++i) {
    allRuntimes[i * 10].reset();
  }
  // now add back 10 runtime instances
  std::vector<std::unique_ptr<Runtime>> newRuntimes;
  for (uint8_t i = 0; i < 10; ++i) {
    newRuntimes.emplace_back(createRuntime());
  }
}
} // namespace
