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

#ifndef TARGET_LIB_CUDA_CONTEXT_H
#define TARGET_LIB_CUDA_CONTEXT_H

#include "Error.h"
#include "HAL/M_driver_memory.h"
#include "Loader/CUDALoader.h"
#include "llvm/ADT/DenseMap.h"

#include <cassert>
#include <cuda.h>

struct M_driver_device;

struct M_driver_context {
  M_driver_device &device;
  CUcontext primaryCtx;

  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  static M_driver_status_t destroy(M_driver_context_t ctx);

  explicit M_driver_context(M_driver_device &dev, CUcontext ctx)
      : device(dev), primaryCtx(ctx) {}

  DriverResult<M_driver_memory_t> allocPinned(uint64_t size);
  DriverResult<M_driver_memory_t> allocSync(uint64_t size);
  DriverResult<M_driver_memory_t> wrapSync(uint64_t address, uint64_t size,
                                           bool owning);
  M_driver_status_t setCurrent();
  M_driver_status_t
  getCurrentDriverContext(M_driver_native_handle_t *driver_context);
  M_driver_status_t
  setCurrentDriverContext(M_driver_native_handle_t driver_context);
  M_driver_status_t memoryInfo(uint64_t *free, uint64_t *total);
  M_driver_status_t enablePeerAccess(M_driver_context &peer);
  M_driver_status_t
  tensorMapEncodeTiled(void *tensor_map, int32_t data_type, uint32_t rank,
                       uint64_t global_device_address,
                       const uint64_t *global_dim,
                       const uint64_t *global_strides, const uint32_t *box_dim,
                       const uint32_t *element_strides, int32_t interleave,
                       int32_t swizzle, int32_t l2_promotion, int32_t oob_fill);
  M_driver_status_t tensorMapEncodeIm2col(
      void *tensor_map, int32_t data_type, uint32_t rank,
      uint64_t global_device_address, const uint64_t *global_dim,
      const uint64_t *global_strides, const int32_t *pixel_box_lower_corner,
      const int32_t *pixel_box_upper_corner, int32_t channels_per_pixel,
      int32_t pixels_per_column, const uint32_t *element_strides,
      int32_t interleave, int32_t swizzle, int32_t l2_promotion,
      int32_t oob_fill);
  M_driver_status_t freePinned(M_driver_memory_t memory);
  M_driver_status_t freeSync(M_driver_memory_t memory);

  M_driver_status_t copyToDeviceSync(M_driver_memory_t dst, uint64_t dstOffset,
                                     const void *src, uint64_t size);
  M_driver_status_t copyFromDeviceSync(void *dst, M_driver_memory_t src,
                                       uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyInterDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size,
                                        M_driver_context_t srcContext);
  M_driver_status_t setMemorySync(M_driver_memory_t dst, uint64_t dstOffset,
                                  uint64_t size, uint8_t value);
  M_driver_status_t fillSync(M_driver_memory_t dst, uint64_t dstOffset,
                             uint64_t size, uint64_t value, uint64_t valueSize);
};

namespace M::Driver::CUDA {

/// RAII guard that sets the CUDA context on construction and restores the
/// previous context on destruction. This is defensive against external
/// libraries (e.g. PyTorch) that may change the current CUDA context.
class ScopedContext {
public:
  M_driver_status_t status;

  ScopedContext(CUcontext ctx, M::Target::CUDA::CUDALoader &cuda)
      : status(M_DRIVER_STATUS_SUCCESS), cuda(cuda), ctx(ctx),
        prevCtx(nullptr) {
    CUresult result = cuda.cuCtxGetCurrent(&prevCtx);
    if (result != CUDA_SUCCESS) {
      status = M_DRIVER_STATUS_BAD_CONTEXT;
      return;
    }
    if (prevCtx == ctx)
      return;
    result = cuda.cuCtxSetCurrent(ctx);
    if (result != CUDA_SUCCESS)
      status = M_DRIVER_STATUS_BAD_CONTEXT;
  }

  ~ScopedContext() {
    if (status != M_DRIVER_STATUS_SUCCESS || prevCtx == ctx)
      return;
    if (prevCtx == nullptr)
      return;
    [[maybe_unused]] CUresult result = cuda.cuCtxSetCurrent(prevCtx);
    assert(result == CUDA_SUCCESS && "Failed to restore previous CUDA context");
  }

  ScopedContext(const ScopedContext &) = delete;
  ScopedContext &operator=(const ScopedContext &) = delete;

private:
  M::Target::CUDA::CUDALoader &cuda;
  CUcontext ctx;
  CUcontext prevCtx;
};

} // namespace M::Driver::CUDA

using M::Driver::CUDA::ScopedContext;

#endif // TARGET_LIB_CUDA_CONTEXT_H
