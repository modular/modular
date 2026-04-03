//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/RuntimeManager.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::AsyncRT;

namespace {

RuntimeRef createRuntime() {
  return AsyncRT::getOrCreateRuntime(AsyncRT::RuntimeSource::Test,
                                     AsyncRT::RuntimeOptions()
                                         .withLeakCheckedAllocator()
                                         .withMainWillNotDonate());
}

/// `getOrCreateRuntime` installs a single global runtime; a second call on the
/// same thread with matching options returns another reference to that runtime.
TEST(RuntimeTest, GetOrCreateRuntimeReturnsSameGlobalInstance) {
  auto first = createRuntime();
  auto second = createRuntime();
  EXPECT_EQ(first.getPointer(), second.getPointer());
}

/// Test to ensure that the thread-local Runtime pointer is cleared when a
/// Runtime is destroyed.
TEST(RuntimeTest, CreateDestroyCreateClearsTls) {
  for (int i = 0; i < 5; ++i) {
    auto runtime = createRuntime();
    runtime.reset(); // Destructor clears thread-local Runtime pointer.
  }
}

TEST(RuntimeTest, DefaultAffinityBehavior) {
  // Ensure env var is not set (may already be in environment)
  unsetenv("MODULAR_ENABLE_AFFINITY");
  AsyncRT::RuntimeOptions options;
  // Disabled by default.
  EXPECT_FALSE(options.withAffinity);
}

TEST(RuntimeTest, EnvVarEnablesAffinity) {
  setenv("MODULAR_ENABLE_AFFINITY", "1", 1);
  AsyncRT::RuntimeOptions options;
  EXPECT_TRUE(options.withAffinity);
  unsetenv("MODULAR_ENABLE_AFFINITY");
}

TEST(RuntimeTest, EnvVarEnablesAffinityWithTrue) {
  setenv("MODULAR_ENABLE_AFFINITY", "true", 1);
  AsyncRT::RuntimeOptions options;
  EXPECT_TRUE(options.withAffinity);
  unsetenv("MODULAR_ENABLE_AFFINITY");
}

TEST(RuntimeTest, BuilderMethodOverridesEnvVar) {
  // Even when env var doesn't enable, builder can enable
  unsetenv("MODULAR_ENABLE_AFFINITY");
  AsyncRT::RuntimeOptions options;
  EXPECT_FALSE(options.withAffinity);
  options.withCPUAffinity(true);
  EXPECT_TRUE(options.withAffinity);
}
} // namespace
