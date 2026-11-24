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
//
// Capture labels on the launch trace: the op span open on the enqueuing thread
// is recorded into each entry while a device-graph capture is in progress, and
// operations on a stream forked into a traced capture land in the owning
// stream's trace. No device needed: streams are opaque handles to the
// registry.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"
#include "Support/Profiling/CaptureLabels.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace M::Driver;
namespace Profiling = M::Profiling;

namespace {

void *fakeStream(uintptr_t value) { return reinterpret_cast<void *>(value); }

KernelLaunchMetadata kernel(const char *name) {
  return KernelLaunchMetadata(name, /*hashValue=*/1);
}

TEST(LaunchTraceLabelsTest, EntriesCarryTheInnermostOp) {
  LaunchTraceRegistry registry;
  void *stream = fakeStream(0x10);
  registry.beginTraceForStream(stream, /*captureId=*/7);

  Profiling::beginCaptureLabeling();
  ASSERT_TRUE(Profiling::pushCaptureLabel("matmul"));
  Profiling::CaptureLabel matmul = Profiling::currentCaptureLabel();
  registry.appendLaunchTraceEntry(stream, kernel("gemm"));
  Profiling::popCaptureLabel();
  registry.appendLaunchTraceEntry(stream, kernel("unlabeled"));
  Profiling::endCaptureLabeling();

  std::vector<KernelLaunchMetadata> trace = registry.takeTraceForStream(stream);
  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0].opSeq, matmul.opSeq);
  EXPECT_EQ(trace[0].opThread, matmul.thread);
  EXPECT_EQ(trace[1].opSeq, 0u);
  EXPECT_EQ(trace[1].opThread, 0u);
}

TEST(LaunchTraceLabelsTest, LabelsDoNotAffectReplayVerification) {
  KernelLaunchMetadata labeled = kernel("gemm");
  labeled.opSeq = 3;
  labeled.opThread = 1;
  EXPECT_EQ(labeled, kernel("gemm"));
}

TEST(LaunchTraceLabelsTest, ForkedStreamResolvesToItsCapture) {
  LaunchTraceRegistry registry;
  void *owner = fakeStream(0x10);
  void *forked = fakeStream(0x20);
  void *unrelated = fakeStream(0x30);
  registry.beginTraceForStream(owner, /*captureId=*/7);

  auto captureIdOf = [&](void *s) -> uint64_t {
    if (s == forked)
      return 7;
    if (s == unrelated)
      return 9;
    return 0;
  };
  EXPECT_EQ(registry.resolveTraceStream(owner, captureIdOf), owner);
  EXPECT_EQ(registry.resolveTraceStream(forked, captureIdOf), owner);
  EXPECT_EQ(registry.resolveTraceStream(unrelated, captureIdOf), nullptr);

  registry.appendLaunchTraceEntry(owner, owner, kernel("main"));
  registry.appendLaunchTraceEntry(owner, forked, kernel("branch"));
  std::vector<KernelLaunchMetadata> trace = registry.takeTraceForStream(owner);
  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0].streamIndex, 0u);
  EXPECT_EQ(trace[1].kernelName, "branch");
  EXPECT_EQ(trace[1].streamIndex, 1u);
}

TEST(LaunchTraceLabelsTest, ForkedStreamResolvesToItsCaptureUnderGlobalTrace) {
  LaunchTraceRegistry registry;
  void *owner = fakeStream(0x10);
  void *forked = fakeStream(0x20);
  void *unrelated = fakeStream(0x30);
  registry.beginGlobalTrace();
  registry.beginTraceForStream(owner, /*captureId=*/7);

  auto captureIdOf = [&](void *s) -> uint64_t { return s == forked ? 7 : 0; };
  EXPECT_EQ(registry.resolveTraceStream(forked, captureIdOf), owner);
  EXPECT_EQ(registry.resolveTraceStream(unrelated, captureIdOf), unrelated);

  registry.appendLaunchTraceEntry(owner, forked, kernel("branch"));
  std::vector<KernelLaunchMetadata> trace = registry.takeTraceForStream(owner);
  ASSERT_EQ(trace.size(), 1u);
  EXPECT_EQ(trace[0].kernelName, "branch");
  EXPECT_EQ(registry.takeGlobalTrace().size(), 1u);
}

TEST(LaunchTraceLabelsTest, NoCaptureMeansNoDriverQuery) {
  LaunchTraceRegistry registry;
  registry.beginTraceForStream(fakeStream(0x10));
  bool queried = false;
  EXPECT_EQ(registry.resolveTraceStream(fakeStream(0x20),
                                        [&](void *) -> uint64_t {
                                          queried = true;
                                          return 0;
                                        }),
            nullptr);
  EXPECT_FALSE(queried);
}

} // namespace
