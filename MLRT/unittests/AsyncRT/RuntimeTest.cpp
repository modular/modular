//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::AsyncRT;

namespace {

RuntimeRef createRuntime() {
  return AsyncRT::createRuntime(AsyncRT::RuntimeSource::Test,
                                AsyncRT::RuntimeOptions()
                                    .withLeakCheckedAllocator()
                                    .withMainWillNotDonate());
}

/// Test to ensure that the thread-local Runtime pointer can only be set once,
// and that it cannot be overwritten by a different Runtime.
#if !defined(NDEBUG)
TEST(RuntimeTest, CreatingSecondRuntimeOnSameThreadAsserts) {
  EXPECT_DEATH(
      {
        auto first = createRuntime();
        (void)first;
        createRuntime(); // Thread already has thread-local Runtime pointer set.
      },
      "creating a runtime from a thread already associated");
}
#endif

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
