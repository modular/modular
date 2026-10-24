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

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "_core/Driver/Device.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace M::Driver::Python {

// Registration is the serial bottleneck (~tens of GiB/s); 16 touch threads
// outrun it while leaving it an uncontended core.
inline constexpr unsigned kDefaultTouchThreads = 16;

// Measured sweet spot across Llama (1 GPU) and Kimi (8 GPU). Smaller pays
// per-call overhead; one huge call exceeds the BAR ceiling and fails with
// CUDA_ERROR_OUT_OF_MEMORY.
inline constexpr size_t kDefaultChunkBytes = 512ULL * 1024 * 1024;

/// One host mapping registered in chunks, because page-locking a
/// multi-hundred-GiB range in one call is slow and impossible above 1 TiB.
///
/// With `driverAlloc`, each chunk is instead its own pinned host allocation
/// from the GPU driver (`Device::chunkAlloc`, bypassing the host memory
/// manager's pool), so the chunks are not contiguous: callers work from
/// `chunkAddresses()`. Pages the driver allocates are not on the kernel's LRU,
/// so memory compaction cannot migrate them out from under the device.
///
/// Not a `Buffer`: a transfer crossing a chunk boundary fails, so callers work
/// from `address()` and keep each transfer inside a row.
struct ChunkedStagingRegion {
  /// Throws on failure, leaving nothing mapped.
  ///
  /// No chunk boundary falls inside a `rowBytes` row.
  ///
  /// REQUIRES: a non-host device whose backend implements `wrapHostMemory`
  /// (with `driverAlloc`, host `chunkAlloc`); `byteSize > 0`, a multiple of
  /// `rowBytes > 0`; `chunkBytes > 0`.
  ChunkedStagingRegion(size_t byteSize, size_t rowBytes, PyDeviceTy device,
                       unsigned threads = kDefaultTouchThreads,
                       size_t chunkBytes = kDefaultChunkBytes,
                       bool driverAlloc = false, bool mlockPages = false);

  /// Drains the device, then unregisters every chunk and unmaps, or frees
  /// every driver-allocated chunk.
  ~ChunkedStagingRegion();

  // The destructor releases, so a copy would double-free.
  ChunkedStagingRegion(const ChunkedStagingRegion &) = delete;
  ChunkedStagingRegion &operator=(const ChunkedStagingRegion &) = delete;
  ChunkedStagingRegion(ChunkedStagingRegion &&other) noexcept;
  ChunkedStagingRegion &operator=(ChunkedStagingRegion &&) = delete;

  /// Valid for `byteSize()` bytes until this is destroyed.
  ///
  /// Throws for a `driverAlloc` region, which has no single base.
  uintptr_t address() const;

  /// Chunk `i` holds bytes `[i * chunkBytes(), (i + 1) * chunkBytes())` of the
  /// region, valid until this is destroyed.
  std::vector<uintptr_t> chunkAddresses() const;

  bool driverAlloc() const { return !driverChunks_.empty(); }

  /// Whether the registered mapping was `mlock`ed.
  bool mlocked() const { return mlocked_; }

  size_t byteSize() const { return byteSize_; }
  size_t rowBytes() const { return rowBytes_; }

  /// The requested `chunkBytes` rounded up to whole rows and pages.
  size_t chunkBytes() const { return chunkBytes_; }

  unsigned numChunks() const {
    return static_cast<unsigned>(driverAlloc() ? driverChunks_.size()
                                               : wraps_.size());
  }

  PyDeviceTy getDevice() const { return device_; }

private:
  PyDeviceTy device_;
  std::vector<M::Driver::DeviceBufferRef> wraps_;
  std::vector<void *> driverChunks_;
  char *base_ = nullptr;
  size_t mapLen_ = 0;
  size_t byteSize_ = 0;
  size_t rowBytes_ = 0;
  size_t chunkBytes_ = 0;
  bool mlocked_ = false;
};

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_CHUNKEDSTAGINGREGION_H
