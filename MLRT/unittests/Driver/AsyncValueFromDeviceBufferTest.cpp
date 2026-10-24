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
// AsyncRT_AsyncValue_createFromDeviceBuffer must adopt a take_handle() handle
// without an addRef. ASAN cannot see a leaked +1, so the tests check the count.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/unittests/Driver/TestUtils.h"

#include "gtest/gtest.h"

#include <cstddef>

using namespace M;
using namespace M::AsyncRT;

// Bare extern "C" export with no header.
extern "C" AsyncValue *
AsyncRT_AsyncValue_createFromDeviceBuffer(Driver::DeviceBuffer *handle);

namespace {

// Creating a "cpu" DeviceContext also installs the process-global CPUDevice
// the entrypoint reads; the returned ref must outlive the call.
void checkAdoptsNetZero(size_t byteSize) {
  ASSIGN_NO_ERR(Driver::DeviceContextRef, ctx,
                Driver::DeviceContext::create(Driver::Device::cpuAPI));
  ASSIGN_NO_ERR(Driver::DeviceBufferRef, bufRef,
                ctx->enqueueCreateBuffer(byteSize));
  ASSERT_TRUE(bufRef->isUnique());

  // Keep-alive copy so the count stays readable after the AsyncValue drops.
  Driver::DeviceBufferRef keepAlive = bufRef.copy();
  Driver::DeviceBuffer *observer = keepAlive.getPointer();
  ASSERT_EQ(observer->getNumReferences(), 2u);

  // Surrender one reference net-zero, as take_handle() does.
  Driver::DeviceBuffer *handle = bufRef.release();
  AnyAsyncValueRef value =
      AnyAsyncValueRef::take(AsyncRT_AsyncValue_createFromDeviceBuffer(handle));

  EXPECT_EQ(observer->getNumReferences(), 2u);
  ASSERT_TRUE(value.isReady());
  EXPECT_EQ(value.get<Driver::DeviceBufferRef>().getPointer(), observer);

  value.reset();
  EXPECT_EQ(observer->getNumReferences(), 1u);
}

TEST(AsyncValueFromDeviceBuffer, AdoptsHandleNetZeroAndReleasesOnDrop) {
  checkAdoptsNetZero(64);
}

TEST(AsyncValueFromDeviceBuffer, ZeroSizeBuffer) { checkAdoptsNetZero(0); }

} // namespace
