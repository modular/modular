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

#include "MLRT/Core/Tensor/TensorBufferRef.h"

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Support/Location.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

static AsyncRT::CPUDeviceRef getCPUDevice() {
  return AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
                                       AsyncRT::CPUDeviceOptions().forDebug());
}

TEST(TensorBufferRef, Default) {
  TensorBufferRef ref;

  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref.getSize(), 0U);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, CreateOwned) {
  auto cpuDevice = getCPUDevice();
  TensorBufferRef ref = TensorBufferRef::createOwned(*cpuDevice, /*size=*/4,
                                                     /*alignment=*/128);
  ASSERT_TRUE(llvm::isAddrAligned(llvm::Align(128), ref.getBuffer()));
  ASSERT_TRUE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, CreateEmptyOwned) {
  auto cpuDevice = getCPUDevice();
  TensorBufferRef ref = TensorBufferRef::createOwned(*cpuDevice, /*size=*/0,
                                                     /*alignment=*/128);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, CreateBorrowed) {
  uint8_t data[4] = {1, 2, 3, 4};
  TensorBufferRef ref =
      TensorBufferRef::createWithNonTrackedMemory(data, /*size=*/4,
                                                  /*alignment=*/{});
  ASSERT_EQ(ref.getSize(), 4u);
  ASSERT_EQ(ref.getBuffer(), data);
}

TEST(TensorBufferRef, CreateEmptyBorrowed) {
  TensorBufferRef ref = TensorBufferRef::createWithNonTrackedMemory(
      /*buffer=*/(void *)nullptr, /*size=*/0,
      /*alignment=*/128);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, Move) {
  auto cpuDevice = getCPUDevice();
  TensorBufferRef ref =
      TensorBufferRef::createOwned(*cpuDevice, sizeof(int), /*alignment=*/16);
  const void *data = ref.getBuffer();

  TensorBufferRef ref2 = std::move(ref);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref2.getBuffer(), data);
}

TEST(TensorBufferRef, Borrow) {
  auto cpuDevice = getCPUDevice();
  TensorBufferRef ref =
      TensorBufferRef::createOwned(*cpuDevice, sizeof(int), /*alignment=*/16);
  TensorBufferRef bor = ref.borrow();
  ASSERT_EQ(bor.getBuffer(), ref.getBuffer());
  ASSERT_EQ(bor.getSize(), ref.getSize());
}

TEST(TensorBufferRef, Destructor) {
  // We are relying on the use of the profiling Allocator in getCPUDevice.
  auto cpuDevice = getCPUDevice();
  void *data = cpuDevice->getAllocator()->allocateBytes(sizeof(int), 16);
  {
    TensorBufferRef ref =
        TensorBufferRef::createWithNonTrackedMemory(data, sizeof(int),
                                                    /*alignment=*/16);
    TensorBufferRef ref2 = std::move(ref);
    // No deallocation.
  }
  cpuDevice->getAllocator()->deallocateBytes(data, sizeof(int));

  {
    TensorBufferRef ref =
        TensorBufferRef::createOwned(*cpuDevice, sizeof(int), /*alignment=*/16);
    TensorBufferRef ref2 = std::move(ref);
    // Will deallocate.
  }
}

TEST(TensorBufferRef, GetMemStorageHandle) {
  auto cpuDevice = getCPUDevice();
  TensorBufferRef *ref =
      new TensorBufferRef(TensorBufferRef::createOwned(*cpuDevice, 4,
                                                       /*alignment=*/{}));
  AnyAsyncValueRef memory = ref->getMemStorageHandle();
  ASSERT_TRUE(memory);

  OwnedMemoryBlock &m = memory.get<OwnedMemoryBlock>();

  ASSERT_EQ(ref->getBuffer(), m.data);

  // This should not delete the underlying memory.
  delete ref;
}
