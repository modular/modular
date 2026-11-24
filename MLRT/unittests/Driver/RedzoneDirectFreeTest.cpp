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
// A redzoned allocation is handed out past its underflow guard. An allocation
// served straight from the driver, not from a pooled chunk, has no descriptor
// that remembers the pointer the driver returned. The invariant under test:
// free gives the driver back the pointer it allocated, not the shifted one.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/MemoryManager/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

using namespace M::Driver;

namespace {

// The CPU device has no memory manager, so the driver serves every allocation.
TEST(RedzoneDirectFreeTest, CpuFreeReturnsUnshiftedPointer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));
  // A skip counts as a pass, so a lost env var must fail the test instead.
  ASSERT_TRUE(ctx->isDebugAllocatorEnabled("out-of-bounds"))
      << "redzone checks are off; set "
         "MODULAR_DEBUG_DEVICE_ALLOCATOR=out-of-bounds";

  // Large enough that tcmalloc serves it from the page heap, which aborts on
  // a pointer that is not the start of an allocation. A small allocation with
  // a shifted pointer is freed without a diagnostic.
  MemoryManagerFreeList &freeList = ctx->freeList();
  ASSIGN_NO_ERR(void *, ptr,
                freeList.allocate(1 << 20, AllocatorContext::AllocMode::SYNC));
  EXPECT_NO_ERR(freeList.free(ptr));
  // The CPU device frees on its work queue, so wait for the free to run.
  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
