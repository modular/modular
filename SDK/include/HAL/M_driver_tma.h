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

#ifndef M_DRIVER_TMA_H
#define M_DRIVER_TMA_H

#include "M_driver_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tensor Memory Accelerator (TMA) descriptor encoding. These mirror the CUDA
// driver `cuTensorMapEncode*` entry points and are only meaningful on backends
// with a TMA engine (NVIDIA Hopper and newer). Backends without one return
// M_DRIVER_STATUS_INCOMPATIBLE. The enum arguments (`data_type`, `interleave`,
// `swizzle`, `l2_promotion`, `oob_fill`) use the same integer encodings as the
// `max.gpu.host` TMA enums, which the plugin maps to the vendor enums.

/**
 * Encode a tiled TMA descriptor into the 128-byte buffer at @p tensor_map.
 *
 * @param[in] context A context, used to identify the plugin and reach its
 *            driver. The descriptor encoders are stateless host-side calls; the
 *            context is not made current.
 * @param[out] tensor_map 128-byte, 64-byte-aligned descriptor storage.
 * @param[in] data_type Element data type (see `_tensormap.DataType`).
 * @param[in] rank Number of tensor dimensions.
 * @param[in] global_device_address Device address of the tensor in global
 *            memory.
 * @param[in] global_dim `rank` global dimension extents.
 * @param[in] global_strides `rank - 1` global strides in bytes (the first
 *            dimension's stride is implicitly the element size).
 * @param[in] box_dim `rank` shared-memory box (tile) extents.
 * @param[in] element_strides `rank` element strides.
 * @param[in] interleave Interleave mode.
 * @param[in] swizzle Swizzle mode.
 * @param[in] l2_promotion L2 promotion mode.
 * @param[in] oob_fill Out-of-bounds fill mode.
 * @return Standard Status Protocol, or M_DRIVER_STATUS_INCOMPATIBLE if the
 *         backend has no TMA engine.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_tensor_map_encode_tiled(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const uint32_t *box_dim,
    const uint32_t *element_strides, int32_t interleave, int32_t swizzle,
    int32_t l2_promotion, int32_t oob_fill);

typedef M_driver_status_t (*M_driver_tensor_map_encode_tiled_func_ptr_t)(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const uint32_t *box_dim,
    const uint32_t *element_strides, int32_t interleave, int32_t swizzle,
    int32_t l2_promotion, int32_t oob_fill);

/**
 * Encode an im2col TMA descriptor into the 128-byte buffer at @p tensor_map.
 *
 * Parameters mirror M_driver_tensor_map_encode_tiled with the box replaced by
 * im2col pixel-box corners and channel/pixel counts.
 *
 * @return Standard Status Protocol, or M_DRIVER_STATUS_INCOMPATIBLE if the
 *         backend has no TMA engine.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_tensor_map_encode_im2col(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const int32_t *pixel_box_lower_corner,
    const int32_t *pixel_box_upper_corner, int32_t channels_per_pixel,
    int32_t pixels_per_column, const uint32_t *element_strides,
    int32_t interleave, int32_t swizzle, int32_t l2_promotion,
    int32_t oob_fill);

typedef M_driver_status_t (*M_driver_tensor_map_encode_im2col_func_ptr_t)(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const int32_t *pixel_box_lower_corner,
    const int32_t *pixel_box_upper_corner, int32_t channels_per_pixel,
    int32_t pixels_per_column, const uint32_t *element_strides,
    int32_t interleave, int32_t swizzle, int32_t l2_promotion,
    int32_t oob_fill);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_TMA_H
