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

#ifndef SDK_DRIVERAPI_PYTHON_CHUNKEDSTAGINGREGION_H
#define SDK_DRIVERAPI_PYTHON_CHUNKEDSTAGINGREGION_H

#include "MLRT/Driver/DeviceContext/ChunkedStagingRegion.h"
#include "_core/Driver/Device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace M::Driver::Python {

inline constexpr unsigned kDefaultTouchThreads =
    M::Driver::kDefaultTouchThreads;
inline constexpr size_t kDefaultChunkBytes = M::Driver::kDefaultChunkBytes;

/// Python wrapper over `M::Driver::ChunkedStagingRegion` that releases the
/// GIL while the region is built and released.
struct ChunkedStagingRegion {
  /// Throws on failure, leaving nothing mapped. See
  /// `M::Driver::ChunkedStagingRegion` for the contract.
  ChunkedStagingRegion(size_t byteSize, size_t rowBytes, PyDeviceTy device,
                       unsigned threads = kDefaultTouchThreads,
                       size_t chunkBytes = kDefaultChunkBytes,
                       bool driverAlloc = false, bool mlockPages = false);

  ~ChunkedStagingRegion();

  ChunkedStagingRegion(const ChunkedStagingRegion &) = delete;
  ChunkedStagingRegion &operator=(const ChunkedStagingRegion &) = delete;
  ChunkedStagingRegion(ChunkedStagingRegion &&other) noexcept = default;
  ChunkedStagingRegion &operator=(ChunkedStagingRegion &&) = delete;

  /// Raises for a `driverAlloc` region, which has no single base.
  uintptr_t address() const;
  std::vector<uintptr_t> chunkAddresses() const {
    return region_->chunkAddresses();
  }
  bool driverAlloc() const { return region_->driverAlloc(); }
  bool mlocked() const { return region_->mlocked(); }
  size_t byteSize() const { return region_->byteSize(); }
  size_t rowBytes() const { return region_->rowBytes(); }
  size_t chunkBytes() const { return region_->chunkBytes(); }
  unsigned numChunks() const { return region_->numChunks(); }
  PyDeviceTy getDevice() const { return device_; }

private:
  PyDeviceTy device_;
  std::unique_ptr<M::Driver::ChunkedStagingRegion> region_;
};

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_CHUNKEDSTAGINGREGION_H
