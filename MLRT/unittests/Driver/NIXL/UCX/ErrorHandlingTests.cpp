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
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace NixlTest;

TEST_F(NixlUcxIntegrationTest, ReceiverFailure) {
  // Test remote disconnection error handling
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  // Use "peer" error handling mode to propagate remote disconnect errors.
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Sender", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1, /*numWorkers=*/1,
                                  /*errorHandlingMode=*/"peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1, /*numWorkers=*/1,
                                  /*errorHandlingMode=*/"peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  const char srcValue = 'S';
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue);

  // Prepare transfer descriptors
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  regSrcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  regDstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // Simulate receiver failure by deregistering memory and destroying agent
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));

  // Destroy the receiver agent to simulate disconnection
  nixl2.agent.reset();

  // Attempt transfer which should fail with NIXL_ERR_REMOTE_DISCONNECT
  transferDataWithFailure(nixl1, nixl2, regSrcDescs, regDstDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, XferThenFail_Write) {
  // Test scenario where first transfer succeeds, then receiver fails before
  // second transfer
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with peer error handling mode.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Sender", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for first transfer.
  size_t transferSize = bufferSize / 2;
  const char srcValue1 = 'A';
  const char srcValue2 = 'B';
  const char initialDstValue = 'Z';

  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue1);
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), initialDstValue);

  // Prepare transfer descriptors.
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  regSrcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  regDstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // First transfer should succeed.
  transferData(nixl1, nixl2, regSrcDescs, regDstDescs);

  // Verify first transfer succeeded.
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0)
      << "First transfer failed to complete correctly";

  // Prepare for second transfer with different data.
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue2);

  // Simulate receiver failure by deregistering memory and destroying agent.
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));
  nixl2.agent.reset();

  // Second transfer should fail with NIXL_ERR_REMOTE_DISCONNECT.
  transferDataWithFailure(nixl1, nixl2, regSrcDescs, regDstDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, XferThenFail_Read) {
  // Test scenario where first READ transfer succeeds, then receiver fails
  // before second transfer
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with peer error handling mode.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Reader", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Source", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for first transfer
  size_t transferSize = bufferSize / 2;
  const char srcValue1 = 'A';
  const char srcValue2 = 'B';
  const char initialDstValue = 'Z';

  std::fill(dramBuffer2.begin(), dramBuffer2.begin() + transferSize, srcValue1);
  std::fill(dramBuffer1.begin(), dramBuffer1.end(), initialDstValue);

  // Prepare transfer descriptors for READ operation
  // For READ: local buffer is destination, remote buffer is source
  nixl_xfer_dlist_t localDescs(nixl_mem_t::DRAM_SEG);
  auto localBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  localDescs.addDesc(basicDescfromArrayRef(localBuf, 0));

  nixl_xfer_dlist_t remoteDescs(nixl_mem_t::DRAM_SEG);
  auto remoteBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  remoteDescs.addDesc(basicDescfromArrayRef(remoteBuf, 0));

  // First READ transfer should succeed.
  transferData(nixl1, nixl2, localDescs, remoteDescs,
               nixl_xfer_op_t::NIXL_READ);

  // Verify first transfer succeeded.
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0)
      << "First READ transfer failed to complete correctly";

  // Prepare for second transfer with different data.
  std::fill(dramBuffer2.begin(), dramBuffer2.begin() + transferSize, srcValue2);

  // Simulate source failure by deregistering memory and destroying agent.
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));
  nixl2.agent.reset();

  // Second READ transfer should fail with NIXL_ERR_REMOTE_DISCONNECT
  transferDataWithFailure(nixl1, nixl2, localDescs, remoteDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, ErrorHandling_PeerMode) {
  // Test explicit peer error handling mode configuration
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create agents with explicit peer error handling mode
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Peer1", available_port(),
                                  "ucx", -1, 1, "peer");

  // Verify that the backend was created with peer mode
  nixl_b_params_t params;
  nixl_mem_list_t mems;
  ASSERT_SUCCESS(nixl1.agent->getBackendParams(nixl1.backend, mems, params));
  ASSERT_EQ(params["ucx_error_handling_mode"], "peer")
      << "Error handling mode was not set to peer";

  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Peer2", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers
  size_t transferSize = bufferSize / 2;
  const char srcValue = 'P';
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue);

  // Prepare descriptors
  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  srcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  dstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // First transfer should succeed with peer mode
  transferData(nixl1, nixl2, srcDescs, dstDescs);

  // Verify transfer
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0)
      << "Transfer with peer mode failed";

  // Now test error handling with peer mode
  // Destroy receiver to trigger peer error handling
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));
  nixl2.agent.reset();

  // Transfer should fail gracefully with peer error handling
  transferDataWithFailure(nixl1, nixl2, srcDescs, dstDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}
