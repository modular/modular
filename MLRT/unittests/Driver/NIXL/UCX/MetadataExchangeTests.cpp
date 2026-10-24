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

TEST_F(NixlUcxIntegrationTest, ListenerCommSize) {
  // Test large metadata exchange to verify segmented message handling
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with listener threads enabled
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_LargeSource",
                                  available_port(), "ucx");

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_LargeDest",
                                  available_port(), "ucx");

  // Register many small buffers to create large metadata
  const size_t bufferSize = 64;
  const size_t numBuffers = 1000;
  std::vector<std::vector<char>> buffers(numBuffers);

  for (size_t i = 0; i < numBuffers; ++i) {
    buffers[i].resize(bufferSize);
    std::fill(buffers[i].begin(), buffers[i].end(),
              static_cast<char>('A' + (i % 26)));
    nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(buffers[i], 0));
  }

  // Exchange metadata - this will create a large message that tests
  // segmentation
  std::string meta1, meta2;
  std::string loadedAgentName1, loadedAgentName2;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));
  ASSERT_SUCCESS(nixl2.agent->getLocalMD(meta2));

  // The large metadata should be exchanged successfully via segmented messages
  ASSERT_SUCCESS(
      nixl1.agent->loadRemoteMD(nixl_blob_t{meta2}, loadedAgentName2));
  ASSERT_EQ(loadedAgentName2, nixl2.agentName);

  // Verify metadata exchange completed within timeout using wait_until_true
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  auto checkBuf = llvm::ArrayRef{buffers[0].data(), bufferSize};
  checkDescs.addDesc(basicDescfromArrayRef(checkBuf, 0));

  ASSERT_TRUE(wait_until_true([&]() {
    return nixl1.agent->checkRemoteMD(nixl2.agentName, checkDescs) ==
           nixl_status_t::NIXL_SUCCESS;
  })) << "Large metadata exchange failed or timed out";
}

TEST_F(NixlUcxIntegrationTest, LoadRemoteWithErrors) {
  // Test error handling when loading metadata fails
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create source agent with backend
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Source", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(1024);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  // Create destination agent WITHOUT backend initially
  std::string destAgentName = "TestAgent_Dest";
  auto nixl2 = std::make_unique<nixlAgent>(
      destAgentName,
      nixlAgentConfig(/*use_prog_thread=*/true, /*use_listen_thread=*/true,
                      /*port=*/available_port()));

  std::string meta1;
  std::string loadedAgentName;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));

  // Try to load metadata without backend - should fail
  ASSERT_NE(nixl2->loadRemoteMD(nixl_blob_t{meta1}, loadedAgentName),
            nixl_status_t::NIXL_SUCCESS);

  // Create backend for destination agent
  nixlBackendH *backend2 = nullptr;
  nixl_b_params_t backendParams;
  nixl_mem_list_t mems;
  // Upstream plugin names are uppercase.
  ASSERT_SUCCESS(nixl2->getPluginParams("UCX", mems, backendParams));
  ASSERT_SUCCESS(nixl2->createBackend("UCX", backendParams, backend2));

  // Try to load invalid metadata - should fail
  std::string invalidMeta = "invalid_metadata_string";
  ASSERT_NE(nixl2->loadRemoteMD(nixl_blob_t{invalidMeta}, loadedAgentName),
            nixl_status_t::NIXL_SUCCESS);

  // Now load valid metadata - should succeed
  ASSERT_SUCCESS(nixl2->loadRemoteMD(nixl_blob_t{meta1}, loadedAgentName));
  ASSERT_EQ(loadedAgentName, nixl1.agentName);
}

TEST_F(NixlUcxIntegrationTest, GetLocalPartialAndLoadRemote) {
  // Test partial metadata exchange functionality
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Partial1", available_port(),
                                  "ucx");

  // Register multiple buffers to test partial metadata
  const size_t bufferSize = 512;
  const size_t numBuffers = 5;
  std::vector<std::vector<char>> buffers1(numBuffers);

  for (size_t i = 0; i < numBuffers; ++i) {
    buffers1[i].resize(bufferSize);
    std::fill(buffers1[i].begin(), buffers1[i].end(), 'A' + i);
    nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(buffers1[i], 0));
  }

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Partial2", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  std::string partialMeta;
  std::string loadedAgentName;

  // Step 1: Get and load connection info only
  nixl_reg_dlist_t emptyList(nixl_mem_t::DRAM_SEG);
  ASSERT_SUCCESS(
      nixl1.agent->getLocalPartialMD(emptyList, partialMeta, nullptr));
  ASSERT_SUCCESS(
      nixl2.agent->loadRemoteMD(nixl_blob_t{partialMeta}, loadedAgentName));
  ASSERT_EQ(loadedAgentName, nixl1.agentName);

  // Verify connection info was loaded
  nixl_xfer_dlist_t emptyXferList(nixl_mem_t::DRAM_SEG);
  ASSERT_SUCCESS(nixl2.agent->checkRemoteMD(nixl1.agentName, emptyXferList));

  // Step 2: Get partial metadata for first 3 buffers only
  nixl_reg_dlist_t partialDescs(nixl_mem_t::DRAM_SEG);
  for (size_t i = 0; i < 3; ++i) {
    partialDescs.addDesc(blobDescfromVector(buffers1[i], 0));
  }

  ASSERT_SUCCESS(
      nixl1.agent->getLocalPartialMD(partialDescs, partialMeta, nullptr));
  ASSERT_SUCCESS(
      nixl2.agent->loadRemoteMD(nixl_blob_t{partialMeta}, loadedAgentName));

  // Verify only the first 3 buffers are available
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  for (size_t i = 0; i < 3; ++i) {
    auto buf = llvm::ArrayRef{buffers1[i].data(), bufferSize};
    checkDescs.addDesc(basicDescfromArrayRef(buf, 0));
    ASSERT_SUCCESS(nixl2.agent->checkRemoteMD(nixl1.agentName, checkDescs));
    checkDescs.clear();
  }

  // Verify buffers 4 and 5 are NOT available
  for (size_t i = 3; i < numBuffers; ++i) {
    nixl_xfer_dlist_t missingDescs(nixl_mem_t::DRAM_SEG);
    auto buf = llvm::ArrayRef{buffers1[i].data(), bufferSize};
    missingDescs.addDesc(basicDescfromArrayRef(buf, 0));
    ASSERT_NE(nixl2.agent->checkRemoteMD(nixl1.agentName, missingDescs),
              nixl_status_t::NIXL_SUCCESS);
  }
}

TEST_F(NixlUcxIntegrationTest, SocketFetchRemote) {
  // Test socket-based metadata fetching
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with listener threads
  size_t bufferSize = 1024;

  NixlAgentWithUCXBackend nixl1;
  int port1 = available_port();
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_FetchSource", port1, "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_FetchDest",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Use fetchRemoteMD to request metadata from nixl1
  nixl_opt_args_t fetchArgs;
  fetchArgs.ipAddr = "127.0.0.1";
  fetchArgs.port = port1;

  ASSERT_SUCCESS(nixl2.agent->fetchRemoteMD(nixl1.agentName, &fetchArgs));

  // Wait for metadata to be received
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  auto checkBuf = llvm::ArrayRef{dramBuffer1.data(), bufferSize};
  checkDescs.addDesc(basicDescfromArrayRef(checkBuf, 0));

  // Use wait_until_true to poll for metadata arrival
  ASSERT_TRUE(wait_until_true([&]() {
    return nixl2.agent->checkRemoteMD(nixl1.agentName, checkDescs) ==
           nixl_status_t::NIXL_SUCCESS;
  })) << "Metadata fetch failed or timed out";

  // Fill source buffer with test data
  const size_t transferSize = bufferSize / 2;
  std::fill(dramBuffer2.begin(), dramBuffer2.begin() + transferSize, 'F');

  // Verify we can create a transfer using the fetched metadata
  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  srcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  dstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // Perform transfer using fetched metadata
  transferData(nixl2, nixl1, srcDescs, dstDescs);

  // Verify transfer succeeded
  ASSERT_EQ(std::memcmp(dramBuffer2.data(), dramBuffer1.data(), transferSize),
            0)
      << "Transfer using fetched metadata failed";
}
