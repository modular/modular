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

#include "Context.h"
#include "Device.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

#include <cuda.h>

using M::Target::CUDA::cudaErrorStatus;

M_driver_status_t M_driver_context::tensorMapEncodeTiled(
    void *tensor_map, int32_t data_type, uint32_t rank,
    uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const uint32_t *box_dim,
    const uint32_t *element_strides, int32_t interleave, int32_t swizzle,
    int32_t l2_promotion, int32_t oob_fill) {
  if (tensor_map == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  // The descriptor encoders are stateless host-side driver calls; reach the
  // device's already-loaded CUDA driver. The max.gpu.host TMA enum encodings
  // match the CUtensorMap* enum values.
  M::Target::CUDA::CUDALoader &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuTensorMapEncodeTiled(
          static_cast<CUtensorMap *>(tensor_map),
          static_cast<CUtensorMapDataType>(data_type), rank,
          reinterpret_cast<void *>(
              static_cast<uintptr_t>(global_device_address)),
          reinterpret_cast<const cuuint64_t *>(global_dim),
          reinterpret_cast<const cuuint64_t *>(global_strides),
          reinterpret_cast<const cuuint32_t *>(box_dim),
          reinterpret_cast<const cuuint32_t *>(element_strides),
          static_cast<CUtensorMapInterleave>(interleave),
          static_cast<CUtensorMapSwizzle>(swizzle),
          static_cast<CUtensorMapL2promotion>(l2_promotion),
          static_cast<CUtensorMapFloatOOBfill>(oob_fill));
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuTensorMapEncodeTiled");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::tensorMapEncodeIm2col(
    void *tensor_map, int32_t data_type, uint32_t rank,
    uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const int32_t *pixel_box_lower_corner,
    const int32_t *pixel_box_upper_corner, int32_t channels_per_pixel,
    int32_t pixels_per_column, const uint32_t *element_strides,
    int32_t interleave, int32_t swizzle, int32_t l2_promotion,
    int32_t oob_fill) {
  if (tensor_map == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  M::Target::CUDA::CUDALoader &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuTensorMapEncodeIm2col(
          static_cast<CUtensorMap *>(tensor_map),
          static_cast<CUtensorMapDataType>(data_type), rank,
          reinterpret_cast<void *>(
              static_cast<uintptr_t>(global_device_address)),
          reinterpret_cast<const cuuint64_t *>(global_dim),
          reinterpret_cast<const cuuint64_t *>(global_strides),
          pixel_box_lower_corner, pixel_box_upper_corner,
          static_cast<cuuint32_t>(channels_per_pixel),
          static_cast<cuuint32_t>(pixels_per_column),
          reinterpret_cast<const cuuint32_t *>(element_strides),
          static_cast<CUtensorMapInterleave>(interleave),
          static_cast<CUtensorMapSwizzle>(swizzle),
          static_cast<CUtensorMapL2promotion>(l2_promotion),
          static_cast<CUtensorMapFloatOOBfill>(oob_fill));
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuTensorMapEncodeIm2col");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_tensor_map_encode_tiled(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const uint32_t *box_dim,
    const uint32_t *element_strides, int32_t interleave, int32_t swizzle,
    int32_t l2_promotion, int32_t oob_fill) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->tensorMapEncodeTiled(
      tensor_map, data_type, rank, global_device_address, global_dim,
      global_strides, box_dim, element_strides, interleave, swizzle,
      l2_promotion, oob_fill);
}

M_driver_status_t M_driver_tensor_map_encode_im2col(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const int32_t *pixel_box_lower_corner,
    const int32_t *pixel_box_upper_corner, int32_t channels_per_pixel,
    int32_t pixels_per_column, const uint32_t *element_strides,
    int32_t interleave, int32_t swizzle, int32_t l2_promotion,
    int32_t oob_fill) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->tensorMapEncodeIm2col(
      tensor_map, data_type, rank, global_device_address, global_dim,
      global_strides, pixel_box_lower_corner, pixel_box_upper_corner,
      channels_per_pixel, pixels_per_column, element_strides, interleave,
      swizzle, l2_promotion, oob_fill);
}
