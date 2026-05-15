//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/HostSystem.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MLRT;

namespace {

CPUDeviceRef createTestCPUDevice() {
  return MLRT::getOrCreateCPUDevice(MLRT::CPUDeviceSource::Test,
                                    MLRT::CPUDeviceOptions()
                                        .withLeakCheckedAllocator()
                                        .withMainWillNotDonate());
}

/// `getOrCreateCPUDevice` installs a single global CPUDevice; a second call on
/// the same thread with matching options returns another reference to that
/// CPUDevice.
TEST(CPUDeviceTest, GetOrCreateCPUDeviceReturnsSameGlobalInstance) {
  auto first = createTestCPUDevice();
  auto second = createTestCPUDevice();
  EXPECT_EQ(first.getPointer(), second.getPointer());
}

/// Test to ensure that the thread-local CPUDevice pointer is cleared when a
/// CPUDevice is destroyed.
TEST(CPUDeviceTest, CreateDestroyCreateClearsTls) {
  for (int i = 0; i < 5; ++i) {
    auto cpuDevice = createTestCPUDevice();
    cpuDevice.reset(); // Destructor clears thread-local CPUDevice pointer.
  }
}

TEST(CPUDeviceTest, DefaultAffinityBehavior) {
  // Ensure env var is not set (may already be in environment)
  unsetenv("MODULAR_ENABLE_AFFINITY");
  MLRT::CPUDeviceOptions options;
  // Disabled by default.
  EXPECT_FALSE(options.withAffinity);
}

TEST(CPUDeviceTest, EnvVarEnablesAffinity) {
  setenv("MODULAR_ENABLE_AFFINITY", "1", 1);
  MLRT::CPUDeviceOptions options;
  EXPECT_TRUE(options.withAffinity);
  unsetenv("MODULAR_ENABLE_AFFINITY");
}

TEST(CPUDeviceTest, EnvVarEnablesAffinityWithTrue) {
  setenv("MODULAR_ENABLE_AFFINITY", "true", 1);
  MLRT::CPUDeviceOptions options;
  EXPECT_TRUE(options.withAffinity);
  unsetenv("MODULAR_ENABLE_AFFINITY");
}

TEST(CPUDeviceTest, BuilderMethodOverridesEnvVar) {
  // Even when env var doesn't enable, builder can enable
  unsetenv("MODULAR_ENABLE_AFFINITY");
  MLRT::CPUDeviceOptions options;
  EXPECT_FALSE(options.withAffinity);
  options.withCPUAffinity(true);
  EXPECT_TRUE(options.withAffinity);
}
} // namespace
