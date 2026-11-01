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

TEST_F(NixlUcxIntegrationTest, UcxThreadpoolSmoke) {
  setenv("UCX_LOG_LEVEL", "warn", 1);

  // Create two agents with UCX backend using a threadpool configuration:
  // 2 workers total, 1 dedicated thread (leaves 1 shared worker).
  const size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TP_Agent1", available_port(), "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/2, /*errMode=*/"",
                                  /*numThreads=*/1);

  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TP_Agent2", available_port(), "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/2, /*errMode=*/"",
                                  /*numThreads=*/1);
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange connection/metadata and perform a small transfer
  exchangeMetadata(nixl1, nixl2);

  const size_t transferSize = bufferSize / 4;
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, 'X');
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), 'Z');

  nixl_xfer_dlist_t src(nixl_mem_t::DRAM_SEG);
  nixl_xfer_dlist_t dst(nixl_mem_t::DRAM_SEG);
  src.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), transferSize}, 0));
  dst.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), transferSize}, 0));

  transferData(nixl1, nixl2, src, dst);

  // Verify copy
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0);
}

TEST_F(NixlUcxIntegrationTest, UcxThreadpoolMultiThread) {
  setenv("UCX_LOG_LEVEL", "warn", 1);

  // Create two agents with UCX backend using a threadpool configuration:
  // 3 workers total, 2 dedicated threads (leaves 1 shared worker).
  // This tests the scenario with numThreads > 1.
  const size_t bufferSize = 4096;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TP_MT_Agent1", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/3, /*errMode=*/"",
                                  /*numThreads=*/2);

  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TP_MT_Agent2", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/3, /*errMode=*/"",
                                  /*numThreads=*/2);
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange connection/metadata
  exchangeMetadata(nixl1, nixl2);

  // Perform multiple transfers to exercise multiple workers
  const size_t transferSize = bufferSize / 2;
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, 'M');
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), 'N');

  nixl_xfer_dlist_t src(nixl_mem_t::DRAM_SEG);
  nixl_xfer_dlist_t dst(nixl_mem_t::DRAM_SEG);
  src.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), transferSize}, 0));
  dst.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), transferSize}, 0));

  transferData(nixl1, nixl2, src, dst);

  // Verify copy
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0);
}

TEST_F(NixlUcxIntegrationTest, MultiWorkerDramTransfer) {
  // Test multi-worker UCX support with concurrent transfers from multiple
  // threads.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Number of workers and threads for testing
  const int numWorkers = 4;
  const int numThreads = 4;
  const size_t bufferSize = 1024;
  const size_t transferSize = bufferSize / 2;

  // Create NIXL agents with multiple workers - use different ports to avoid
  // conflicts
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_MultiWorker1", 8047, "ucx",
                                  -1, numWorkers);

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_MultiWorker2", 8052, "ucx",
                                  -1, numWorkers);

  // Register multiple buffers for concurrent transfers
  std::vector<std::vector<char>> dramBuffers1(numThreads);
  std::vector<std::vector<char>> dramBuffers2(numThreads);

  for (int i = 0; i < numThreads; ++i) {
    dramBuffers1[i].resize(bufferSize);
    dramBuffers2[i].resize(bufferSize);

    // Fill source buffers with unique values
    std::fill(dramBuffers1[i].begin(), dramBuffers1[i].end(), 'A' + i);
    std::fill(dramBuffers2[i].begin(), dramBuffers2[i].end(), 'Z');

    // Register memory for each buffer
    nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(dramBuffers1[i], 0));
    nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(dramBuffers2[i], 0));
  }

  // Exchange metadata
  exchangeMetadata(nixl1, nixl2);

  // Launch concurrent transfers from multiple threads
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([&, i]() {
      // Prepare transfer descriptors for this thread
      nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
      auto srcBuf = llvm::ArrayRef{dramBuffers1[i].data(), transferSize};
      srcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

      nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
      auto dstBuf = llvm::ArrayRef{dramBuffers2[i].data(), transferSize};
      dstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

      // Create and execute transfer
      nixlXferReqH *reqHandle = nullptr;
      nixl_opt_args_t xferArgs;
      xferArgs.notifMsg = "notification_thread_" + std::to_string(i);
      xferArgs.hasNotif = true;

      if (nixl1.agent->createXferReq(
              nixl_xfer_op_t::NIXL_WRITE, srcDescs, dstDescs, nixl2.agentName,
              reqHandle, &xferArgs) == nixl_status_t::NIXL_SUCCESS) {

        // Post the transfer
        nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
        if (xferStatus == nixl_status_t::NIXL_SUCCESS ||
            xferStatus == nixl_status_t::NIXL_IN_PROG) {

          // Poll for completion
          nixl_status_t checkStatus = xferStatus;
          const auto startTime = std::chrono::steady_clock::now();
          const auto timeout = std::chrono::seconds(10);

          while (checkStatus == nixl_status_t::NIXL_IN_PROG &&
                 (std::chrono::steady_clock::now() - startTime < timeout)) {
            checkStatus = nixl1.agent->getXferStatus(reqHandle);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }

          if (checkStatus == nixl_status_t::NIXL_SUCCESS) {
            // Verify the transfer
            if (std::memcmp(dramBuffers1[i].data(), dramBuffers2[i].data(),
                            transferSize) == 0) {
              ++successCount;
            }
          }

          nixl1.agent->releaseXferReq(reqHandle);
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Verify all transfers completed successfully
  ASSERT_EQ(successCount, numThreads)
      << "Not all concurrent transfers completed successfully. "
      << "Success count: " << successCount << "/" << numThreads;

  // Additionally verify all destination buffers have correct data
  for (int i = 0; i < numThreads; ++i) {
    ASSERT_EQ(std::memcmp(dramBuffers1[i].data(), dramBuffers2[i].data(),
                          transferSize),
              0)
        << "Buffer " << i << " transfer verification failed";

    // Check untouched region
    ASSERT_EQ(dramBuffers2[i][transferSize], 'Z')
        << "Buffer " << i << " has data corruption beyond transfer region";
  }
}

TEST_F(NixlUcxIntegrationTest, SelfNotification) {
  // Test self notification support
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create a single NIXL agent that will send notifications to itself.
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Self", available_port(),
                                  "ucx");

  // Test multiple self notifications.
  const int numNotifications = 10;
  std::vector<std::string> sentMessages;

  for (int i = 0; i < numNotifications; ++i) {
    std::string msg = "self_notification_" + std::to_string(i);
    sentMessages.push_back(msg);
    ASSERT_SUCCESS(nixl1.agent->genNotif(nixl1.agentName, msg));
  }

  // Poll for notifications with timeout.
  nixl_notifs_t notifMap;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);
  size_t receivedCount = 0;

  while (receivedCount < numNotifications &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    ASSERT_SUCCESS(nixl1.agent->getNotifs(notifMap));
    if (notifMap.count(nixl1.agentName) > 0) {
      receivedCount = notifMap[nixl1.agentName].size();
    }
    if (receivedCount < numNotifications) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Verify we received all self notifications.
  ASSERT_EQ(receivedCount, numNotifications)
      << "Did not receive all self notifications within timeout";

  // Verify notification content matches what was sent.
  ASSERT_EQ(notifMap.count(nixl1.agentName), 1)
      << "Self notification map missing agent";

  const std::vector<nixl_blob_t> &receivedMessages = notifMap[nixl1.agentName];
  ASSERT_EQ(receivedMessages.size(), sentMessages.size())
      << "Incorrect number of self notifications received";

  // Check that all sent messages were received (order may vary).
  for (const std::string &sentMsg : sentMessages) {
    auto it =
        std::find(receivedMessages.begin(), receivedMessages.end(), sentMsg);
    ASSERT_NE(it, receivedMessages.end())
        << "Self notification message not found: " << sentMsg;
  }
}

TEST_F(NixlUcxIntegrationTest, ConcurrentNotifications) {
  // Test concurrent genNotif calls with RW sync mode.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents configured with RW sync mode.
  NixlAgentWithUCXBackend nixl1;
  nixl1.agentName = "TestAgent_NotifSender";
  nixl1.agent = std::make_unique<nixlAgent>(
      nixl1.agentName,
      nixlAgentConfig(/*use_prog_thread=*/true, /*use_listen_thread=*/true,
                      /*port=*/available_port(),
                      /*sync_mode=*/nixl_thread_sync_t::NIXL_THREAD_SYNC_RW));

  // Create backend with UCX.
  nixl_b_params_t backendParams;
  nixl_mem_list_t mems;
  std::vector<nixl_backend_t> plugins;
  ASSERT_SUCCESS(nixl1.agent->getAvailPlugins(plugins));
  // Upstream plugin names are uppercase; use case-insensitive check for
  // defence-in-depth, but pass the canonical uppercase name to the API.
  bool ucxFound =
      std::find_if(plugins.begin(), plugins.end(), [](const nixl_backend_t &p) {
        return llvm::StringRef(p).lower() == "ucx";
      }) != plugins.end();
  ASSERT_TRUE(ucxFound) << "UCX plugin not found";

  ASSERT_SUCCESS(nixl1.agent->getPluginParams("UCX", mems, backendParams));
  ASSERT_SUCCESS(
      nixl1.agent->createBackend("UCX", backendParams, nixl1.backend));

  // Register memory.
  std::vector<char> dramBuffer1(1024);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  // Create second agent with RW sync mode.
  NixlAgentWithUCXBackend nixl2;
  nixl2.agentName = "TestAgent_NotifReceiver";
  nixl2.agent = std::make_unique<nixlAgent>(
      nixl2.agentName,
      nixlAgentConfig(/*use_prog_thread=*/true, /*use_listen_thread=*/true,
                      /*port=*/available_port(),
                      /*sync_mode=*/nixl_thread_sync_t::NIXL_THREAD_SYNC_RW));

  ASSERT_SUCCESS(nixl2.agent->getPluginParams("UCX", mems, backendParams));
  ASSERT_SUCCESS(
      nixl2.agent->createBackend("UCX", backendParams, nixl2.backend));

  std::vector<char> dramBuffer2(1024);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Launch concurrent notification threads.
  const int numThreads = 4;
  const int notifsPerThread = 10;
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int threadId = 0; threadId < numThreads; ++threadId) {
    threads.emplace_back([&, threadId]() {
      for (int i = 0; i < notifsPerThread; ++i) {
        std::string msg =
            "thread" + std::to_string(threadId) + "_notif" + std::to_string(i);
        auto status = nixl1.agent->genNotif(nixl2.agentName, msg);
        if (status == nixl_status_t::NIXL_SUCCESS)
          ++successCount;
      }
    });
  }

  // Wait for all threads to complete.
  for (std::thread &thread : threads)
    thread.join();

  // Verify all notifications were sent successfully
  ASSERT_EQ(successCount, numThreads * notifsPerThread)
      << "Not all concurrent notifications were sent successfully";

  // Poll for notifications with timeout.
  nixl_notifs_t notifMap;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);
  size_t receivedCount = 0;

  while (receivedCount < static_cast<size_t>(numThreads * notifsPerThread) &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    ASSERT_SUCCESS(nixl2.agent->getNotifs(notifMap));
    if (notifMap.count(nixl1.agentName) > 0) {
      receivedCount = notifMap[nixl1.agentName].size();
    }
    if (receivedCount < static_cast<size_t>(numThreads * notifsPerThread)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Verify we received all notifications.
  ASSERT_EQ(receivedCount, static_cast<size_t>(numThreads * notifsPerThread))
      << "Did not receive all concurrent notifications within timeout";
}
