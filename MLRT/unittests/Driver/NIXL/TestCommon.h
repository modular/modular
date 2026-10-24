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

#ifndef MLRT_UNITTESTS_DRIVER_NIXL_TESTCOMMON_H
#define MLRT_UNITTESTS_DRIVER_NIXL_TESTCOMMON_H

#include "nixl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

std::ostream &operator<<(std::ostream &os, nixl_status_t status);

/// Sets NIXL_PLUGIN_DIR from the Bazel runfiles tree so the upstream plugin
/// manager can find libplugin_UCX.so / libplugin_LIBFABRIC.so. Call once
/// from a GTest environment or from main() before constructing any nixlAgent.
/// No-op if NIXL_PLUGIN_DIR is already set in the environment.
void configureNixlPluginDir();

#define ASSERT_SUCCESS(EXPR)                                                   \
  do {                                                                         \
    nixl_status_t status = (EXPR);                                             \
    ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS) << #EXPR;                   \
  } while (false)

namespace NixlTest {

template <class T>
nixlBlobDesc blobDescfromVector(const std::vector<T> &vec, uint64_t devId,
                                llvm::StringRef metaInfo = "") {
  return {reinterpret_cast<uintptr_t>(vec.data()), vec.size() * sizeof(T),
          devId, metaInfo.str()};
}

template <class T>
nixlBasicDesc basicDescfromArrayRef(llvm::ArrayRef<T> ref, uint64_t devId) {
  return {reinterpret_cast<uintptr_t>(ref.data()), ref.size() * sizeof(T),
          devId};
}

int available_port(int startPort = 8000, int endPort = 9000,
                   int maxAttempts = 100);

bool wait_until_true(std::function<bool()> func, int retries = 500);

struct NixlAgentWithUCXBackend {
  std::string agentName;
  std::unique_ptr<nixlAgent> agent;
  nixlBackendH *backend = nullptr;

  void registerMemory(nixl_mem_t memType, nixlBlobDesc desc);

  static void create(NixlAgentWithUCXBackend &self, std::string agentName,
                     int port, std::string backendName, int gpuDeviceId = -1,
                     int numWorkers = 1, std::string errorHandlingMode = "",
                     int numThreads = 0);
};

void exchangeMetadata(NixlAgentWithUCXBackend &nixl1,
                      NixlAgentWithUCXBackend &nixl2);

void transferData(NixlAgentWithUCXBackend &nixl1,
                  NixlAgentWithUCXBackend &nixl2, nixl_xfer_dlist_t &srcDescs,
                  nixl_xfer_dlist_t &dstDescs,
                  nixl_xfer_op_t op = nixl_xfer_op_t::NIXL_WRITE);

void transferDataWithFailure(NixlAgentWithUCXBackend &nixl1,
                             NixlAgentWithUCXBackend &nixl2,
                             nixl_xfer_dlist_t &srcDescs,
                             nixl_xfer_dlist_t &dstDescs,
                             nixl_status_t expectedStatus);

class NixlUcxIntegrationTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { configureNixlPluginDir(); }
};

} // namespace NixlTest

#endif // MLRT_UNITTESTS_DRIVER_NIXL_TESTCOMMON_H
