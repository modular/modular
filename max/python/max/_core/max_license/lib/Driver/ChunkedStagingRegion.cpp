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

#include "_core/Driver/ChunkedStagingRegion.h"
#include "nanobind/nanobind.h"

#include <stdexcept>
#include <utility>

using namespace M;
using namespace Driver;
namespace nb = nanobind;

Python::ChunkedStagingRegion::ChunkedStagingRegion(
    size_t byteSize, size_t rowBytes, PyDeviceTy device, unsigned threads,
    size_t chunkBytes, bool driverAlloc, bool mlockPages)
    : device_(std::move(device)) {
  if (device_->isHost())
    throw std::invalid_argument(
        "ChunkedStagingRegion requires a non-host device "
        "(GPU/Accelerator).");
  M::AsyncRT::CPUDeviceRef cpu = Python::getCpuDevice(0)->getAsyncRTCPUDevice();
  M::Driver::ChunkedStagingRegionError error;
  {
    nb::gil_scoped_release nogil;
    region_ = M::Driver::ChunkedStagingRegion::create(
        device_->ctx, *cpu, byteSize, rowBytes, threads, chunkBytes,
        driverAlloc, mlockPages, error);
  }
  if (!region_) {
    if (error.invalidArgument)
      throw std::invalid_argument(error.message);
    throw std::runtime_error(error.message);
  }
}

uintptr_t Python::ChunkedStagingRegion::address() const {
  if (driverAlloc())
    throw std::runtime_error("ChunkedStagingRegion: a driver-allocated "
                             "region has no single base; use "
                             "chunk_addresses");
  return region_->address();
}

Python::ChunkedStagingRegion::~ChunkedStagingRegion() {
  if (!region_)
    return;
  // Releasing drains the device, which blocks.
  nb::gil_scoped_release nogil;
  region_.reset();
}
