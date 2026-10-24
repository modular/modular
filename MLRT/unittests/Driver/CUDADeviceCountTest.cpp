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
// Counting CUDA devices must not initialize the CUDA driver. This lives in its
// own binary because the invariant is a property of a process that has not
// touched CUDA yet, and any other test in the same binary would have called
// cuInit first.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/NVML.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <dlfcn.h>

using namespace M::Driver;
using M::ErrorOr;

namespace {

// Whether cuInit has run in this process.
//
// Library residency is not the question: loading libnvidia-ml maps libcuda
// alongside it, so it is resident either way. What does not survive a fork is
// the driver state cuInit builds, so the probe is a call that reports whether
// that state exists — cuDeviceGetCount answers CUDA_ERROR_NOT_INITIALIZED
// until it does, and asking does not create it.
bool isCUDAInitialized() {
  constexpr unsigned int CUDA_ERROR_NOT_INITIALIZED = 3;

  void *handle = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW);
  if (!handle)
    return false;

  auto *cuDeviceGetCount = reinterpret_cast<unsigned int (*)(int *)>(
      dlsym(handle, "cuDeviceGetCount"));
  int count = 0;
  bool initialized = cuDeviceGetCount &&
                     cuDeviceGetCount(&count) != CUDA_ERROR_NOT_INITIALIZED;

  dlclose(handle);
  return initialized;
}

// Whether counting is entitled to reach for CUDA — see
// nvmlCountMayDisagreeWithCUDA in CUDADeviceContext.cpp. Duplicated rather than
// exported because a test that shared the predicate with the code under test
// would pass no matter which way that predicate went.
bool migInPlay(const NVML &nvml) {
  unsigned int nvmlCount;
  if (nvml.nvmlDeviceGetCount_v2(&nvmlCount) != NVML::SUCCESS)
    return true;

  for (unsigned int i = 0; i < nvmlCount; ++i) {
    nvmlDevice_t device;
    if (nvml.nvmlDeviceGetHandleByIndex(i, &device) != NVML::SUCCESS)
      return true;

    unsigned int currentMode, pendingMode;
    NVML::Result result =
        nvml.nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);
    if (result == NVML::NOT_SUPPORTED)
      continue;
    if (result != NVML::SUCCESS || currentMode != NVML_DEVICE_MIG_DISABLE)
      return true;
  }
  return false;
}

// Callers that only want a device count — accelerator_count() in the Python
// driver is the one that matters — must not pay cuInit for it. A process that
// has initialized CUDA breaks a CUDA-using child forked from it afterwards,
// which is how importing MAX alongside vLLM once made vLLM unable to start.
TEST(CUDADeviceCountTest, CountingDevicesDoesNotInitializeCUDA) {
  ASSERT_FALSE(isCUDAInitialized())
      << "the CUDA driver was already initialized before the first count, so "
         "this test cannot observe what counting initializes; it must run in "
         "its own binary";

  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();
  if (migInPlay(*nvmlOr))
    GTEST_SKIP() << "MIG is enabled, where only CUDA can count the devices a "
                    "container was granted";

  ASSIGN_NO_ERR(int, count, CUDADevice::numberOfGPUs());
  EXPECT_GT(count, 0);

  EXPECT_FALSE(isCUDAInitialized())
      << "counting " << count
      << " devices initialized the CUDA driver; NVML answers this on hardware "
         "without MIG, and cuInit here breaks a CUDA-using child forked later";
}

} // namespace
