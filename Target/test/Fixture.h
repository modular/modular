//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_api.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::cerr;
using std::endl;

// Macro for interim lines in a test.
// These build up to the items being tested but
// themselves should have been previously tested.
// So this construct hides their `EXPECT_*()` so that
// the test code only shows items of interest.
#define RUN(stmt)                                                              \
  do {                                                                         \
    M_driver_status_t status = (this->stmt);                                   \
    ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS);                                \
  } while (0)

#define DO_TEST(stmt)                                                          \
  do {                                                                         \
    M_driver_status_t status = (this->stmt);                                   \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS) << getStatusMessage(status);  \
  } while (0)

// The ASSERT_SUCCESS helper macro should be used when calling driver entry
// points when the expected result is M_DRIVER_STATUS_SUCCESS and execution
// should NOT continue if the call fails. This is should be preferred over
// EXPECT_SUCCESS by default.
#define ASSERT_SUCCESS(EXPR)                                                   \
  do {                                                                         \
    M_driver_status_t status = (EXPR);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS) << getStatusMessage(status);  \
  } while (0)

// The EXPECT_SUCCESS helper macro should be used when calling driver entry
// points when the expected result is M_DRIVER_STATUS_SUCCESS and test
// execution should continue if the call fails. Only use instead of
// ASSERT_SUCCESS if there is resource cleanup or other processing required
// after the calling the driver entrypoint.
#define EXPECT_SUCCESS(EXPR)                                                   \
  do {                                                                         \
    M_driver_status_t status = (EXPR);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS) << getStatusMessage(status);  \
  } while (0)

// Some plugins take seconds to create a context. With
// `MODULAR_DRIVER_TEST_SHARE_CONTEXTS=1`, consecutive context-level tests
// borrow one driver, its devices and their contexts from this session instead
// of creating them per test. Any other test drops the session first, so a test
// that creates contexts itself never runs while shared ones are alive. Unset,
// every test creates and destroys its own.
struct SharedPluginSession {
  void *soHandle = nullptr;
  M_driver_destroy_func_ptr_t destroy = nullptr;
  M_driver_context_destroy_func_ptr_t context_destroy = nullptr;
  M_driver_driver_t driver = nullptr;
  bool hasDevices = false;
  std::vector<M_driver_device_t> devices;
  bool hasContexts = false;
  std::vector<M_driver_context_t> contexts;

  static bool enabled() {
    static const bool on = [] {
      const char *env = std::getenv("MODULAR_DRIVER_TEST_SHARE_CONTEXTS");
      return env != nullptr && std::string(env) == "1";
    }();
    return on;
  }

  static std::unique_ptr<SharedPluginSession> &current() {
    static std::unique_ptr<SharedPluginSession> session;
    return session;
  }

  // Destroys the shared contexts and driver, if there are any.
  static void drop() {
    std::unique_ptr<SharedPluginSession> &session = current();
    if (!session)
      return;
    for (M_driver_context_t context : session->contexts)
      if (context != nullptr)
        session->context_destroy(context);
    if (session->driver != nullptr)
      session->destroy(session->driver);
    dlclose(session->soHandle);
    session.reset();
  }
};

struct DriverPluginTest : ::testing::Test {
  void SetUp() override {
    const char *pluginSpec = std::getenv("MODULAR_DRIVER_PLUGINS");
    if (pluginSpec == nullptr) {
      GTEST_SKIP() << "MODULAR_DRIVER_PLUGINS environment variable is not set";
      return;
    }
    llvm::StringRef soPath(pluginSpec);
    if (soPath.empty()) {
      GTEST_SKIP() << "MODULAR_DRIVER_PLUGINS environment variable is not set";
      return;
    }
    soHandle = dlopen(pluginSpec, RTLD_NOW);
    ASSERT_TRUE(soHandle != nullptr);

#define M_DRIVER_HAL_SYMBOL(name)                                              \
  name = (M_driver_##name##_func_ptr_t)dlsym(                                  \
      soHandle, ("M_driver_" + std::string(#name)).c_str());                   \
  ASSERT_TRUE(name != nullptr) << "Failed to resolve M_driver_" #name;

#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
    M_driver_version version = {.major = M_DRIVER_INTERFACE_VERSION_MAJOR,
                                .minor = M_DRIVER_INTERFACE_VERSION_MINOR,
                                .patch = M_DRIVER_INTERFACE_VERSION_PATCH};
    if (sharesContexts && SharedPluginSession::enabled()) {
      std::unique_ptr<SharedPluginSession> &session =
          SharedPluginSession::current();
      if (!session) {
        session = std::make_unique<SharedPluginSession>();
        // Its own reference keeps the plugin loaded between tests.
        session->soHandle = dlopen(pluginSpec, RTLD_NOW);
        ASSERT_TRUE(session->soHandle != nullptr);
        session->destroy = destroy;
        session->context_destroy = context_destroy;
        DO_TEST(create(&version, &session->driver));
      }
      driver = session->driver;
      usingSharedSession = true;
    } else {
      SharedPluginSession::drop();
      DO_TEST(create(&version, &driver));
    }
    ASSERT_EQ(version.major, M_DRIVER_INTERFACE_VERSION_MAJOR);
    ASSERT_EQ(version.minor, M_DRIVER_INTERFACE_VERSION_MINOR);

    const char *apiStr = nullptr;
    M_driver_status_t status = property(driver, M_DRIVER_PROPERTY_API, &apiStr);
    if (status == M_DRIVER_STATUS_SUCCESS && apiStr)
      api = apiStr;
  }

  void TearDown() override {
    if (soHandle == nullptr)
      return;
    if (!usingSharedSession)
      DO_TEST(destroy(driver));
#define M_DRIVER_HAL_SYMBOL(name) name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
    dlclose(soHandle);
  }

  std::string getStatusMessage(M_driver_status_t status) {
    char errorMessage[1024];
    M_driver_status_t msgStatus = this->status_message(
        driver, status, errorMessage, sizeof(errorMessage));
    if (msgStatus != M_DRIVER_STATUS_SUCCESS)
      return std::string("Failed to get status message");
    return std::string(errorMessage);
  }

  // Some plugins take seconds to create a context per device, which adds up
  // on a host with dozens of cards. Set `MODULAR_DRIVER_TEST_MAX_DEVICES` to
  // test only the first N. A missing or malformed value tests every device,
  // so a typo never quietly turns the per-device tests off.
  static int64_t testDeviceCount(int64_t deviceCount) {
    static const int64_t cap = [] {
      const char *env = std::getenv("MODULAR_DRIVER_TEST_MAX_DEVICES");
      if (env == nullptr)
        return int64_t{0};
      char *end = nullptr;
      long long value = std::strtoll(env, &end, 10);
      return end != env && *end == '\0' ? static_cast<int64_t>(value)
                                        : int64_t{0};
    }();
    return cap > 0 ? std::min(deviceCount, cap) : deviceCount;
  }

  void *soHandle = nullptr;
#define M_DRIVER_HAL_SYMBOL(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
  M_driver_driver_t driver = nullptr;

  llvm::StringRef api;

  // Set by the context-level fixtures, whose tests may borrow the shared
  // session.
  bool sharesContexts = false;
  bool usingSharedSession = false;
};

struct DriverPluginDeviceTest : DriverPluginTest {
  void SetUp() override {
    DriverPluginTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    SharedPluginSession *session =
        usingSharedSession ? SharedPluginSession::current().get() : nullptr;
    if (session != nullptr && session->hasDevices) {
      devices = session->devices;
      return;
    }
    int64_t deviceCount = 0;
    ASSERT_SUCCESS(device_count(driver, &deviceCount));
    deviceCount = testDeviceCount(deviceCount);
    for (int64_t i = 0; i < deviceCount; i++) {
      M_driver_device_t device = nullptr;
      EXPECT_SUCCESS(device_get(driver, i, &device));
      devices.push_back(device);
    }
    if (session != nullptr) {
      session->devices = devices;
      session->hasDevices = true;
    }
  }

  std::vector<M_driver_device_t> devices;
};

struct DriverPluginContextTest : DriverPluginDeviceTest {
  DriverPluginContextTest() { sharesContexts = true; }

  void SetUp() override {
    DriverPluginDeviceTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    SharedPluginSession *session =
        usingSharedSession ? SharedPluginSession::current().get() : nullptr;
    if (session != nullptr && session->hasContexts) {
      contexts = session->contexts;
      return;
    }
    for (M_driver_device_t device : devices) {
      M_driver_context_t context = nullptr;
      ASSERT_SUCCESS(context_create(device, &context));
      contexts.push_back(context);
    }
    if (session != nullptr) {
      session->contexts = contexts;
      session->hasContexts = true;
    }
  }

  void TearDown() override {
    if (usingSharedSession) {
      // A test may have destroyed and recreated one; hand back what it holds.
      if (SharedPluginSession *session = SharedPluginSession::current().get())
        session->contexts = contexts;
    } else {
      for (M_driver_context_t context : contexts) {
        EXPECT_SUCCESS(context_destroy(context));
      }
    }
    DriverPluginTest::TearDown();
  }

  std::vector<M_driver_context_t> contexts;
};

struct DriverPluginQueueTest : DriverPluginContextTest {
  void SetUp() override {
    DriverPluginContextTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (M_driver_context_t context : contexts) {
      M_driver_queue_t queue;
      ASSERT_SUCCESS(queue_create(context, &queue));
      queues.push_back(queue);
    }
  }

  void TearDown() override {
    for (size_t i = 0; i < queues.size(); i++)
      EXPECT_SUCCESS(queue_destroy(contexts[i], queues[i]));
    DriverPluginContextTest::TearDown();
  }

  std::vector<M_driver_queue_t> queues;
};
