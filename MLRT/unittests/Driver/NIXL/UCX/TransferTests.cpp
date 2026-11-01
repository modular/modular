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

#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace NixlTest;

TEST_F(NixlUcxIntegrationTest, DramTransfer) {
  // Explicitly enable verbose logging for UCX.
  // This is incredibly helpful to leave enabled by default.
  // If a test passes locally but fails in CI, it lets us see logs in CI.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents on separate ports with 1KB host buffers
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Ucx1", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Ucx2", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  size_t srcOffset = 0;
  size_t dstOffset = bufferSize / 4; // Arbitrary offset

  ASSERT_LE(srcOffset + transferSize, dramBuffer1.size());
  ASSERT_LE(dstOffset + transferSize, dramBuffer2.size());

  const char srcValue = 'S';
  const char initialDstValue = 'D';

  std::fill(dramBuffer1.begin() + srcOffset,
            dramBuffer1.begin() + srcOffset + transferSize, srcValue);
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), initialDstValue);

  // === Prepare Transfer Descriptors ===
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data() + srcOffset, transferSize};
  regSrcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data() + dstOffset, transferSize};
  regDstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  transferData(nixl1, nixl2, regSrcDescs, regDstDescs);

  // Verify transferred data
  ASSERT_EQ(std::memcmp(dramBuffer1.data() + srcOffset,
                        dramBuffer2.data() + dstOffset, transferSize),
            0)
      << "Transferred data in dramBuffer2 does not match source in dramBuffer1";

  // Check if data outside the destination range was untouched
  if (dstOffset > 0) {
    ASSERT_EQ(dramBuffer2[dstOffset - 1], initialDstValue)
        << "Data before destination offset was modified";
  }
  if (dstOffset + transferSize < dramBuffer2.size()) {
    ASSERT_EQ(dramBuffer2[dstOffset + transferSize], initialDstValue)
        << "Data after destination range was modified";
  }
}

TEST_F(NixlUcxIntegrationTest, BasicXfer_Read) {
  // Test basic successful READ operation.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Reader", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Source", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  size_t srcOffset = 0;
  size_t dstOffset = bufferSize / 4;

  const char srcValue = 'S';
  const char initialDstValue = 'D';

  std::fill(dramBuffer2.begin() + srcOffset,
            dramBuffer2.begin() + srcOffset + transferSize, srcValue);
  std::fill(dramBuffer1.begin(), dramBuffer1.end(), initialDstValue);

  // Prepare transfer descriptors for READ operation
  nixl_xfer_dlist_t localDescs(nixl_mem_t::DRAM_SEG);
  auto localBuf = llvm::ArrayRef{dramBuffer1.data() + dstOffset, transferSize};
  localDescs.addDesc(basicDescfromArrayRef(localBuf, 0));

  nixl_xfer_dlist_t remoteDescs(nixl_mem_t::DRAM_SEG);
  auto remoteBuf = llvm::ArrayRef{dramBuffer2.data() + srcOffset, transferSize};
  remoteDescs.addDesc(basicDescfromArrayRef(remoteBuf, 0));

  // Perform READ transfer
  transferData(nixl1, nixl2, localDescs, remoteDescs,
               nixl_xfer_op_t::NIXL_READ);

  // Verify transferred data
  ASSERT_EQ(std::memcmp(dramBuffer1.data() + dstOffset,
                        dramBuffer2.data() + srcOffset, transferSize),
            0)
      << "READ: Transferred data in dramBuffer1 does not match source in "
         "dramBuffer2";

  // Check if data outside the destination range was untouched
  if (dstOffset > 0) {
    ASSERT_EQ(dramBuffer1[dstOffset - 1], initialDstValue)
        << "READ: Data before destination offset was modified";
  }
  if (dstOffset + transferSize < dramBuffer1.size()) {
    ASSERT_EQ(dramBuffer1[dstOffset + transferSize], initialDstValue)
        << "READ: Data after destination range was modified";
  }
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryDisabled) {
  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("NIXL_TELEMETRY_ENABLE", "n", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Telemetry1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Telemetry2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  EXPECT_EQ(nixl1.agent->getXferTelemetry(reqHandle, telemetry),
            nixl_status_t::NIXL_ERR_NO_TELEMETRY);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  unsetenv("NIXL_TELEMETRY_ENABLE");
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryEnabledByEnv) {
  namespace fs = std::filesystem;

  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("NIXL_TELEMETRY_ENABLE", "y", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_ucx_telem_env_" + std::to_string(getpid()) + "_" +
       std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);
  setenv("NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_TelemetryEnabled1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_TelemetryEnabled2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  ASSERT_SUCCESS(nixl1.agent->getXferTelemetry(reqHandle, telemetry));
  EXPECT_EQ(telemetry.totalBytes, bufferSize);
  EXPECT_EQ(telemetry.descCount, 1u);
  EXPECT_GE(telemetry.postDuration, chrono_period_us_t(0));
  EXPECT_GE(telemetry.xferDuration, telemetry.postDuration);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  ASSERT_SUCCESS(nixl1.agent->invalidateRemoteMD(nixl2.agentName));
  ASSERT_SUCCESS(nixl2.agent->invalidateRemoteMD(nixl1.agentName));
  nixl_reg_dlist_t regDescs1(nixl_mem_t::DRAM_SEG);
  regDescs1.addDesc(blobDescfromVector(dramBuffer1, 0));
  nixl_opt_args_t regArgs1;
  regArgs1.backends = {nixl1.backend};
  ASSERT_SUCCESS(nixl1.agent->deregisterMem(regDescs1, &regArgs1));
  nixl_reg_dlist_t regDescs2(nixl_mem_t::DRAM_SEG);
  regDescs2.addDesc(blobDescfromVector(dramBuffer2, 0));
  nixl_opt_args_t regArgs2;
  regArgs2.backends = {nixl2.backend};
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(regDescs2, &regArgs2));
  fs::remove_all(telemetryDir);
  unsetenv("NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("NIXL_TELEMETRY_DIR");
  unsetenv("NIXL_TELEMETRY_ENABLE");
}

// Pins the contract that ``xferDuration`` reflects when the backend's
// progress thread observed the transfer complete, NOT the wall-clock at
// which the agent's poll loop happened to call ``getXferStatus``.
//
// The test forces a large gap between actual completion and the first
// poll: it waits for the receiver's notification to land (proxy for
// "the transfer is done"), sleeps 50ms, *then* makes its first
// ``getXferStatus`` call. With the old poll-time stamping, xferDuration
// would inflate to 50ms+. With ``getCompletionTime`` plumbed through
// it stays bounded by the actual transfer.
TEST_F(NixlUcxIntegrationTest, XferDurationDoesNotInflateUnderPollDelay) {
  namespace fs = std::filesystem;

  setenv("NIXL_TELEMETRY_ENABLE", "y", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_ucx_telem_progress_" + std::to_string(getpid()) + "_" +
       std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);
  setenv("NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_XferDurationBounded1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_XferDurationBounded2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs;
  xferArgs.notifMsg = "progress_thread_completion";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);

  // Wait for the receiver to see the notification: by that time the
  // sender's progress thread has also flipped the request to SUCCESS.
  // We poll getXferStatus on nixl1 to drive progress on both sides (upstream
  // v1.1.0 requires explicit polling to flush the UCX send queue; pure
  // progress-thread delivery is not guaranteed for loopback AM notifications).
  // The poll drives progress WITHOUT recording the poll-delay in xferDuration:
  // upstream stamped xferDuration in the progress thread when the backend
  // completed, so our poll time after the fact does not inflate it.
  ASSERT_TRUE(wait_until_true([&]() {
    // Drive nixl1's send-side completion (required for notification delivery).
    nixl1.agent->getXferStatus(reqHandle);
    nixl_notifs_t notifMap;
    if (nixl2.agent->getNotifs(notifMap) != nixl_status_t::NIXL_SUCCESS) {
      return false;
    }
    auto it = notifMap.find(nixl1.agentName);
    return it != notifMap.end() && !it->second.empty();
  }));

  // Inject a polling gap. xferDuration must not absorb this delay because
  // the backend stamped it when the transfer completed, not when we polled.
  constexpr auto kPollGap = std::chrono::milliseconds(50);
  const auto deadline = std::chrono::steady_clock::now() + kPollGap;
  while (std::chrono::steady_clock::now() < deadline) {
  }

  // Confirm the transfer is complete (progress-thread already set it).
  status = nixl1.agent->getXferStatus(reqHandle);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  ASSERT_SUCCESS(nixl1.agent->getXferTelemetry(reqHandle, telemetry));
  // xferDuration should reflect when the progress thread completed the
  // transfer, NOT when we polled after the 50ms gap.
  EXPECT_LT(telemetry.xferDuration, kPollGap)
      << "xferDuration=" << telemetry.xferDuration.count()
      << "us; expected to reflect progress-thread completion, not the "
      << kPollGap.count() << "ms poll delay.";

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  ASSERT_SUCCESS(nixl1.agent->invalidateRemoteMD(nixl2.agentName));
  ASSERT_SUCCESS(nixl2.agent->invalidateRemoteMD(nixl1.agentName));
  nixl_reg_dlist_t regDescs1(nixl_mem_t::DRAM_SEG);
  regDescs1.addDesc(blobDescfromVector(dramBuffer1, 0));
  nixl_opt_args_t regArgs1;
  regArgs1.backends = {nixl1.backend};
  ASSERT_SUCCESS(nixl1.agent->deregisterMem(regDescs1, &regArgs1));
  nixl_reg_dlist_t regDescs2(nixl_mem_t::DRAM_SEG);
  regDescs2.addDesc(blobDescfromVector(dramBuffer2, 0));
  nixl_opt_args_t regArgs2;
  regArgs2.backends = {nixl2.backend};
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(regDescs2, &regArgs2));
  fs::remove_all(telemetryDir);
  unsetenv("NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("NIXL_TELEMETRY_DIR");
  unsetenv("NIXL_TELEMETRY_ENABLE");
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryEnabledForMakeXferReq) {
  namespace fs = std::filesystem;

  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("NIXL_TELEMETRY_ENABLE", "y", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_ucx_telem_make_req_" + std::to_string(getpid()) + "_" +
       std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);
  setenv("NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_TelemetryMakeReq1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_TelemetryMakeReq2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlDlistH *localHandle = nullptr;
  nixlDlistH *remoteHandle = nullptr;
  ASSERT_SUCCESS(nixl1.agent->prepXferDlist("", srcDescs, localHandle));
  ASSERT_SUCCESS(
      nixl1.agent->prepXferDlist(nixl2.agentName, dstDescs, remoteHandle));

  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->makeXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                          localHandle, {0}, remoteHandle, {0},
                                          reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  ASSERT_SUCCESS(nixl1.agent->getXferTelemetry(reqHandle, telemetry));
  EXPECT_EQ(telemetry.totalBytes, bufferSize);
  EXPECT_EQ(telemetry.descCount, 1u);
  EXPECT_GE(telemetry.postDuration, chrono_period_us_t(0));
  EXPECT_GE(telemetry.xferDuration, telemetry.postDuration);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  ASSERT_SUCCESS(nixl1.agent->releasedDlistH(localHandle));
  ASSERT_SUCCESS(nixl1.agent->releasedDlistH(remoteHandle));
  ASSERT_SUCCESS(nixl1.agent->invalidateRemoteMD(nixl2.agentName));
  ASSERT_SUCCESS(nixl2.agent->invalidateRemoteMD(nixl1.agentName));
  nixl_reg_dlist_t regDescs1(nixl_mem_t::DRAM_SEG);
  regDescs1.addDesc(blobDescfromVector(dramBuffer1, 0));
  nixl_opt_args_t regArgs1;
  regArgs1.backends = {nixl1.backend};
  ASSERT_SUCCESS(nixl1.agent->deregisterMem(regDescs1, &regArgs1));
  nixl_reg_dlist_t regDescs2(nixl_mem_t::DRAM_SEG);
  regDescs2.addDesc(blobDescfromVector(dramBuffer2, 0));
  nixl_opt_args_t regArgs2;
  regArgs2.backends = {nixl2.backend};
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(regDescs2, &regArgs2));
  fs::remove_all(telemetryDir);
  unsetenv("NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("NIXL_TELEMETRY_DIR");
  unsetenv("NIXL_TELEMETRY_ENABLE");
}
