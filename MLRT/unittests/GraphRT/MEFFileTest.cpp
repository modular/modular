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

#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveDescriptor.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/Buffer.h"
#include "Support/ML/SizeUtils.h"
#include "Support/RCRef.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>

using namespace M;
using namespace M::AsyncRT;
using bazel::tools::cpp::runfiles::Runfiles;

namespace {
class TrackedBuffer final : public Buffer {
public:
  explicit TrackedBuffer(std::unique_ptr<llvm::MemoryBuffer> storage)
      : Buffer(std::move(storage)) {}
  ~TrackedBuffer() override { destroyed.set_value(); }

  std::promise<void> destroyed;
};

class MEFFileTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
    ASSERT_NE(runfiles, nullptr) << error;
    auto input = llvm::MemoryBuffer::getFile(runfiles->Rlocation(
        "_main/MLRT/unittests/GraphRT/buffer_lifetime.mef"));
    ASSERT_TRUE(input) << input.getError().message();
    auto aligned = createAndCopyToAlignedBuffer((*input)->getBufferStart(),
                                                (*input)->getBufferEnd());
    auto trackedBuffer = RCRef<TrackedBuffer>::create(std::move(aligned));
    destroyed = trackedBuffer->destroyed.get_future();
    buffer = std::move(trackedBuffer);

    registry.add("test.function", [](RunnablePrimitiveDescriptor &) {});
    registry.add("test.increment", [](RunnablePrimitiveDescriptor &frame) {
      frame.emplaceResult<int32_t>(0, frame.getArg<int32_t>(0) + 1);
    });
    registry.finalize();
  }

  void open(bool installAtOpen) {
    auto fileOr = installAtOpen ? MEFFile::open(buffer.copy(), registry)
                                : MEFFile::open(buffer.copy());
    ASSERT_FALSE(fileOr.isError());
    file = fileOr.takeValue();
    if (!installAtOpen)
      ASSERT_FALSE(file->installPrimitives(registry).isError());
    auto error = file->getTopLevelInvokable()->walkPrimitives(
        [&](StringRef name, const PrimitiveDescriptor &desc) -> ErrorOrSuccess {
          if (name == "test.function")
            invokable = desc.getInvokable(0);
          return success();
        });
    ASSERT_FALSE(error.isError());
    ASSERT_NE(invokable, nullptr);
  }

  std::future<void> destroyed;
  PrimitiveRegistry registry;
  BufferRef buffer;
  RCRef<MEFFile> file;
  MEFInvokable *invokable = nullptr;
};

TEST_F(MEFFileTest, PendingExecutionRetainsBuffer) {
  ASSERT_NO_FATAL_FAILURE(open(true));
  auto cpuDevice = getOrCreateCPUDevice(
      CPUDeviceSource::Test, CPUDeviceOptions().withMainWillNotDonate());
  auto input = AsyncValueRef<int32_t>::allocate(cpuDevice.getPointer());
  AnyAsyncValueRef arguments[] = {input.copy()};
  auto results = invokable->invoke(arguments, *cpuDevice);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_FALSE(results[0].isReady());

  // Only the pending executor now owns the file and its serialized users list.
  arguments[0].reset();
  buffer.reset();
  file.reset();
  ASSERT_EQ(destroyed.wait_for(std::chrono::seconds(0)),
            std::future_status::timeout);

  std::move(input).emplace(41);
  AsyncRT::await(results);
  // Output readiness can precede executor teardown. Wait for destruction
  // separately rather than treating await as a quiescence barrier.
  ASSERT_TRUE(results[0].isReady());
  ASSERT_FALSE(results[0].isError());
  EXPECT_EQ(results[0].get<int32_t>(), 42);
  EXPECT_EQ(destroyed.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

TEST_F(MEFFileTest, RetainedInvokableRetainsBuffer) {
  ASSERT_NO_FATAL_FAILURE(open(false));
  auto retainedInvokable = copyRCRef(invokable);
  buffer.reset();
  file.reset();
  ASSERT_EQ(destroyed.wait_for(std::chrono::seconds(0)),
            std::future_status::timeout);

  auto cpuDevice = getOrCreateCPUDevice(
      CPUDeviceSource::Test, CPUDeviceOptions().withMainWillNotDonate());
  AnyAsyncValueRef arguments[] = {
      AsyncValueRef<int32_t>::createReady(cpuDevice.getPointer(), 41)};
  auto results = retainedInvokable->invoke(arguments, *cpuDevice);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].isReady());
  ASSERT_FALSE(results[0].isError());
  EXPECT_EQ(results[0].get<int32_t>(), 42);
  EXPECT_EQ(destroyed.wait_for(std::chrono::seconds(0)),
            std::future_status::timeout);
  retainedInvokable.reset();
  EXPECT_EQ(destroyed.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}
} // namespace
