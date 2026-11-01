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

#include "MLRT/unittests/Driver/NIXL/TestCommon.h"

#include "Support/NixlPluginDir.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <netinet/in.h>
#include <random>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

std::ostream &operator<<(std::ostream &os, nixl_status_t status) {
  return os << nixlEnumStrings::statusStr(status);
}

// Applies the per-vendor flavor resolution (Support/NixlPluginDir.h) to a
// candidate @nixl_upstream root and sets NIXL_PLUGIN_DIR when a flavor
// resolves. Returns true when the candidate root existed (whether or not a
// flavor was found), ending the search.
static bool setPluginDirFromRepoRoot(const std::filesystem::path &root) {
  std::error_code ec;
  if (!std::filesystem::exists(root, ec))
    return false;
  // Force the plain (non-verbs) flavor: these loopback tests assert intra-host
  // latencies, such as xferDuration staying under a 50 ms poll gap, that the
  // verbs flavor's IB endpoint wireup can exceed. The verbs/rc path is
  // validated cross-process by the cross-node smoke instead; see
  // resolveNixlPluginDir's allowVerbsFlavor note.
  if (auto vendorDir =
          M::resolveNixlPluginDir(root, /*allowVerbsFlavor=*/false))
    setenv("NIXL_PLUGIN_DIR", vendorDir->c_str(), /*overwrite=*/0);
  return true;
}

void configureNixlPluginDir() {
  // If the caller already set NIXL_PLUGIN_DIR, respect it.
  if (std::getenv("NIXL_PLUGIN_DIR"))
    return;

  // Pre-load the GPU runtime libraries with RTLD_LAZY|RTLD_GLOBAL so their
  // symbols are visible when nixlPluginManager calls
  //   dlopen(libplugin_UCX.so, RTLD_NOW)
  // libplugin_UCX.so is built with -Wl,-z,undefs, leaving CUDA driver/NVML
  // symbols (cuda flavor) or HSA symbols (rocm flavors) unresolved. RTLD_NOW at
  // dlopen time requires ALL symbols to be resolvable from currently loaded
  // libraries. Pre-loading these with RTLD_GLOBAL makes them available. The
  // rdma-core entries matter only to the *-verbs flavors, which these tests do
  // not load.
  // Failures are non-fatal: on CPU-only machines the plugin simply won't load.
  const char *gpuLibs[] = {
      "libcuda.so.1",          "libcuda.so",
      "libnvidia-ml.so.1",     "libnvidia-ml.so",
      "libhsa-runtime64.so.1", "libhsa-runtime64.so",
      "libibverbs.so.1",       "libmlx5.so.1",
  };
  for (const char *lib : gpuLibs) {
    dlopen(lib, RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
  }

  // The upstream nixl plugin manager looks for libplugin_UCX.so and
  // libplugin_LIBFABRIC.so in NIXL_PLUGIN_DIR. Under Bazel tests the plugins
  // are runfiles in per-vendor subdirectories of
  // <runfiles_root>/+http_archive+nixl_upstream/.
  //
  // Strategy 1: Bazel sets TEST_SRCDIR to the runfiles root. Use it directly.
  if (const char *testSrcDir = std::getenv("TEST_SRCDIR")) {
    if (setPluginDirFromRepoRoot(std::filesystem::path(testSrcDir) /
                                 "+http_archive+nixl_upstream"))
      return;
  }

  // Strategy 2: Walk up the directory tree from the binary looking for the
  // "+http_archive+nixl_upstream" directory (handles remote build environments
  // where the binary lives inside .runfiles/_main/...).
  std::error_code ec;
  auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec)
    return;

  auto path = exe.parent_path();
  while (!path.empty() && path != path.parent_path()) {
    if (setPluginDirFromRepoRoot(path / "+http_archive+nixl_upstream"))
      return;
    path = path.parent_path();
  }

  // Strategy 3: classic <binary>.runfiles layout (non-test-wrapper use).
  setPluginDirFromRepoRoot(exe.string() +
                           ".runfiles/+http_archive+nixl_upstream");
}

namespace {
// GTest global environment: calls configureNixlPluginDir() in SetUp(), which
// runs before any test suite fixture and therefore before any nixlAgent
// constructor (which triggers the nixlPluginManager Meyers-singleton). GTest's
// global environment SetUp() is the earliest reliable hook available in the
// LLVM TestMain.cpp test harness.
class NixlPluginDirEnv : public ::testing::Environment {
public:
  void SetUp() override { configureNixlPluginDir(); }
};

// Registered here so every translation unit that links TestCommon gets the
// environment. ::testing::AddGlobalTestEnvironment takes ownership.
// NOLINTNEXTLINE: global constructor is intentional — we need to register
// before main() runs (actually before GTest's Run() — GTest environments are
// invoked in registration order before any test suite).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
const bool kNixlPluginDirEnvRegistered = [] {
  ::testing::AddGlobalTestEnvironment(new NixlPluginDirEnv);
  return true;
}();
#pragma clang diagnostic pop
} // namespace

namespace NixlTest {

void NixlAgentWithUCXBackend::registerMemory(nixl_mem_t memType,
                                             nixlBlobDesc desc) {
  nixl_reg_dlist_t descList(memType);
  descList.addDesc(desc);
  nixl_opt_args_t regArgs;
  regArgs.backends = {backend};
  ASSERT_SUCCESS(agent->registerMem(descList, &regArgs));
}

void NixlAgentWithUCXBackend::create(NixlAgentWithUCXBackend &self,
                                     std::string agentName, int port,
                                     std::string backendName, int gpuDeviceId,
                                     int numWorkers,
                                     std::string errorHandlingMode,
                                     int numThreads) {
  self.agentName = agentName;
  self.agent = std::make_unique<nixlAgent>(
      agentName, nixlAgentConfig(/*use_prog_thread=*/true,
                                 /*use_listen_thread=*/true,
                                 /*port=*/port));
  std::vector<nixl_backend_t> plugins;
  ASSERT_SUCCESS(self.agent->getAvailPlugins(plugins));
  // Upstream plugin names are uppercase (UCX, LIBFABRIC). Do a
  // case-insensitive match so callers can pass either form, and capture the
  // canonical name (uppercase) returned by the plugin manager to use in
  // subsequent getPluginParams / createBackend calls.
  nixl_backend_t canonicalName;
  auto it = std::find_if(plugins.begin(), plugins.end(),
                         [&](const nixl_backend_t &p) {
                           return llvm::StringRef(p).lower() ==
                                  llvm::StringRef(backendName).lower();
                         });
  ASSERT_NE(it, plugins.end()) << "Plugin not found: " << backendName;
  canonicalName = *it;

  nixl_b_params_t backendParams;
  nixl_mem_list_t mems;
  ASSERT_SUCCESS(
      self.agent->getPluginParams(canonicalName, mems, backendParams));

  if (gpuDeviceId != -1) {
    backendParams["gpu_device_id"] = std::to_string(gpuDeviceId);
  }
  if (numWorkers > 1) {
    backendParams["num_workers"] = std::to_string(numWorkers);
  }
  if (!errorHandlingMode.empty()) {
    backendParams["ucx_error_handling_mode"] = errorHandlingMode;
  }
  if (numThreads > 0) {
    backendParams["num_threads"] = std::to_string(numThreads);
  }

  ASSERT_SUCCESS(
      self.agent->createBackend(canonicalName, backendParams, self.backend));
  ASSERT_NE(self.backend, nullptr) << "Backend handle is null";
}

int available_port(int startPort, int endPort, int maxAttempts) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(startPort, endPort);

  for (int i = 0; i < maxAttempts; ++i) {
    int port = dis(gen);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      continue;
    }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      close(sock);
      return port;
    }

    close(sock);
  }

  llvm_unreachable("no available port found");
}

bool wait_until_true(std::function<bool()> func, int retries) {
  bool result;
  while (!(result = func()) && retries-- > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return result;
}

void exchangeMetadata(NixlAgentWithUCXBackend &nixl1,
                      NixlAgentWithUCXBackend &nixl2) {
  std::string meta1, meta2;
  std::string loadedAgentName1, loadedAgentName2;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));
  ASSERT_SUCCESS(nixl2.agent->getLocalMD(meta2));

  ASSERT_FALSE(meta1.empty()) << "Agent 1 metadata is empty";
  ASSERT_FALSE(meta2.empty()) << "Agent 2 metadata is empty";

  ASSERT_SUCCESS(
      nixl1.agent->loadRemoteMD(nixl_blob_t{meta2}, loadedAgentName2));
  ASSERT_EQ(loadedAgentName2, nixl2.agentName);

  ASSERT_SUCCESS(
      nixl2.agent->loadRemoteMD(nixl_blob_t{meta1}, loadedAgentName1));
  ASSERT_EQ(loadedAgentName1, nixl1.agentName);
}

void transferData(NixlAgentWithUCXBackend &nixl1,
                  NixlAgentWithUCXBackend &nixl2, nixl_xfer_dlist_t &srcDescs,
                  nixl_xfer_dlist_t &dstDescs, nixl_xfer_op_t op) {
  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(
      op, srcDescs, dstDescs, nixl2.agentName, reqHandle, &xferArgs));

  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";
  nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
  if (xferStatus != nixl_status_t::NIXL_SUCCESS &&
      xferStatus != nixl_status_t::NIXL_IN_PROG) {
    FAIL() << "postXferReq failed: " << nixlEnumStrings::statusStr(xferStatus);
  }

  nixl_notifs_t notifMap;
  nixl_status_t checkStatus = nixl_status_t::NIXL_IN_PROG;
  bool notifReceived = false;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);

  while ((checkStatus != nixl_status_t::NIXL_SUCCESS || !notifReceived) &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    if (checkStatus != nixl_status_t::NIXL_SUCCESS) {
      checkStatus = nixl1.agent->getXferStatus(reqHandle);
      ASSERT_NE(checkStatus, nixl_status_t::NIXL_ERR_UNKNOWN)
          << "getXferStatus returned error";
    }

    if (!notifReceived) {
      ASSERT_SUCCESS(nixl2.agent->getNotifs(notifMap));
      if (notifMap.count(nixl1.agentName) > 0 &&
          !notifMap[nixl1.agentName].empty()) {
        notifReceived = true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  ASSERT_EQ(checkStatus, nixl_status_t::NIXL_SUCCESS)
      << "Transfer did not complete within timeout";
  ASSERT_TRUE(notifReceived) << "Notification not received within timeout";
  ASSERT_EQ(notifMap.count(nixl1.agentName), 1)
      << "Notification map missing agent1";
  ASSERT_EQ(notifMap[nixl1.agentName].size(), 1)
      << "Incorrect number of notifications received";
}

void transferDataWithFailure(NixlAgentWithUCXBackend &nixl1,
                             NixlAgentWithUCXBackend &nixl2,
                             nixl_xfer_dlist_t &srcDescs,
                             nixl_xfer_dlist_t &dstDescs,
                             nixl_status_t expectedStatus) {
  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));

  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";
  nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
  if (xferStatus != expectedStatus &&
      xferStatus != nixl_status_t::NIXL_IN_PROG) {
    FAIL() << "postXferReq returned unexpected status: "
           << nixlEnumStrings::statusStr(xferStatus);
  }

  nixl_status_t checkStatus = xferStatus;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(30);

  while (checkStatus == nixl_status_t::NIXL_IN_PROG &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    checkStatus = nixl1.agent->getXferStatus(reqHandle);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  ASSERT_EQ(checkStatus, expectedStatus)
      << "Transfer did not complete with expected status. Got: "
      << nixlEnumStrings::statusStr(checkStatus)
      << ", Expected: " << nixlEnumStrings::statusStr(expectedStatus);

  nixl1.agent->releaseXferReq(reqHandle);
}

} // namespace NixlTest
