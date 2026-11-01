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

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "Support/TypeID.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

struct PayloadA : public ReferenceCounted<PayloadA> {
  explicit PayloadA(int i) : i(i) { resource = malloc(1); }
  ~PayloadA() { free(resource); }
  int i;
  void *resource;
};

struct PayloadB : public ReferenceCounted<PayloadB> {
  explicit PayloadB(char c) : c(c) {}
  char c;
};

static AsyncRT::CPUDeviceRef getCPUDevice() {
  return AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
                                       AsyncRT::CPUDeviceOptions().forDebug());
}

#ifndef NDEBUG
TEST(ContextTest, StateSlotDowncastFailTest) {
  StateContext ctx(/*numSlots=*/1);
  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, 43);
  ASSERT_DEATH_IF_SUPPORTED(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/0),
                            "mismatch between actual and expected type ids");
}
#endif

TEST(ContextTest, StateSlotTest) {
  StateContext ctx(/*numSlots=*/2);
  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, 43);
  ctx.setStateSlot(/*slotIndex=*/1, RCRef<PayloadB>::create('a'));

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadA>(/*slotIndex=*/0).i, 43);

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/1).c, 'a');
}

TEST(ContextTest, getOrCreateBufferTest) {
  auto cpuDevice = getCPUDevice();
  {
    // Add scope to make sure that the allocator outlives the context.
    StateContext ctx(/*numSlots=*/1);
    auto &tensorSlot = ctx.emplaceStateSlot<TensorStateSlot>(/*slotIndex=*/0);
    {
      auto buffer = tensorSlot.getOrCreateBuffer(*cpuDevice, /*size=*/1,
                                                 /*alignment=*/{});

      // getOrCreateBuffer call with the same slot and the same size returns a
      // cached buffer.
      auto buffer2 = tensorSlot.getOrCreateBuffer(*cpuDevice, /*size=*/1,
                                                  /*alignment=*/{});
      EXPECT_EQ(buffer.getBuffer(), buffer2.getBuffer());

      // We're done with the above buffers, so safe to reallocate.
    }

    // Call with a different size will allocate a new buffer.
    auto buffer3 = tensorSlot.getOrCreateBuffer(*cpuDevice, /*size=*/2,
                                                /*alignment=*/{});
    EXPECT_NE(buffer3.getSize(), 1UL);
  }
}

TEST(ContextTest, setZeroSizedBufferTest) {
  auto cpuDevice = getCPUDevice();
  {
    // Add scope to make sure that the allocator outlives the context.
    StateContext ctx(/*numSlots=*/1);
    auto &tensorSlot = ctx.emplaceStateSlot<TensorStateSlot>(/*slotIndex=*/0);

    // A zero-sized buffer may have a null data pointer while still tracking
    // its storage.
    TensorBufferRef owner =
        TensorBufferRef::createOwned(*cpuDevice, /*size=*/1, /*alignment=*/{});
    ErrorOrSuccess setOr = tensorSlot.setBuffer(TensorBufferRef::create(
        /*buffer=*/nullptr, /*size=*/0, owner.getMemStorageHandle()));
    ASSERT_FALSE(setOr.isError());

    EXPECT_FALSE(tensorSlot.getBufferRef().isError());
  }
}
