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
// C entry to the decode dispatch of the attention kernels.
//
// A captured device graph freezes the launch its decode attention kernels
// pick from host scalars, so a serving engine keys its captures on that
// launch. `M_attnDispatchResolve` returns it from the kernels' own code, for
// every attention family, so the engine holds no copy of the dispatch logic.
//
// The header is plain C.
//
//===----------------------------------------------------------------------===//

#ifndef KERNELS_ATTN_DISPATCH_H
#define KERNELS_ATTN_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define M_ATTN_DISPATCH_API __declspec(dllexport)
#else
#define M_ATTN_DISPATCH_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Value of a descriptor field its attention family does not set. Zero is a
/// valid window and a valid stride, so it cannot mean "unset".
#define M_ATTN_DISPATCH_UNSET (-1)

/// Most scalars a key or a metadata record holds.
#define M_ATTN_DISPATCH_MAX_SCALARS 8

/// Values of `MAttnDispatchDesc::kv_dtype`. New values are appended only.
enum {
  M_ATTN_KV_DTYPE_OTHER = 0,
  M_ATTN_KV_DTYPE_FP8 = 1,
};

/// Bits of `MAttnDispatchDesc::flags`. New bits are appended only.
enum {
  /// The cache's key is wider than its value.
  M_ATTN_FLAG_SPLIT_HEAD_DIMS = 1u << 0,
  /// A sparse decode passes neither per-token top-k lengths nor a sink.
  M_ATTN_FLAG_FIXED_TOPK = 1u << 1,
  /// Sparse rows are physical.
  M_ATTN_FLAG_PHYSICAL_INDICES = 1u << 2,
  /// The launch is the row-invariant pin's own verify.
  M_ATTN_FLAG_ROW_INVARIANT_VERIFY = 1u << 3,
};

/// The facts of one KV cache that a decode dispatch policy reads: the union
/// over every attention family. A policy ignores the fields it does not read
/// and fails on a field it needs that is `M_ATTN_DISPATCH_UNSET`, and on a
/// head count or a page size of zero, so a caller cannot omit one or
/// zero-initialize the struct and get a wrong key.
///
/// What the device knows (its API and its multiprocessor count) is not here:
/// the policy reads it from the device context, the source the launch uses.
///
/// New fields are appended only. A field past the caller's `struct_size` reads
/// as unset, so a caller built against an older header stays valid.
typedef struct MAttnDispatchDesc {
  /// `sizeof(MAttnDispatchDesc)` in the caller's build.
  uint32_t struct_size;
  /// The attention family: "mha", "mla", "msa", "msa_index".
  const char *kind;
  /// Query heads on one device.
  int64_t num_q_heads;
  /// KV heads on one device.
  int64_t num_kv_heads;
  /// The cache row width the dispatch assumes.
  int64_t head_dim;
  int64_t page_size;
  /// The attention window; unset when there is none.
  int64_t window_size;
  /// The sparse index row stride; unset for a dense cache.
  int64_t sparse_indices_stride;
  /// An `M_ATTN_KV_DTYPE_*` value.
  int32_t kv_dtype;
  /// The sparse MLA cache form (0 plain, 1 scaled FP8, 2 NVFP4); unset when
  /// it does not apply.
  int32_t kv_form;
  /// A bitset of `M_ATTN_FLAG_*`.
  uint32_t flags;
} MAttnDispatchDesc;

/// What one decode dispatch resolves to.
typedef struct MAttnDispatch {
  /// The part of the capture identity the kernels pick: the split count,
  /// then the launch variant, each present only when the family picks one.
  /// A caller adds the batch size and the query width.
  int64_t key[M_ATTN_DISPATCH_MAX_SCALARS];
  int32_t n_key;
  /// The scalars staged into the dispatch record. `n_metadata` is 0 when no
  /// kernel reads one.
  int64_t metadata[M_ATTN_DISPATCH_MAX_SCALARS];
  int32_t n_metadata;
} MAttnDispatch;

/// Resolves the decode dispatch of one KV cache at one step shape.
///
/// A caller that captures at probed cache lengths finds where the key
/// changes between two of them by bisecting this function. Nothing promises
/// that a key holds over one range of cache lengths only.
///
/// @param device_ctx The context of the device the decode runs on, as
///   `M_DriverDeviceContextHandle` returns it.
/// @param desc The cache's descriptor.
/// @param batch_size Requests in the decode batch.
/// @param q_max_seq_len Query rows per request: 1 for plain decode.
/// @param max_cache_valid_length The batch's largest valid cache length.
/// @param out Receives the dispatch on success.
/// @param err On failure receives a message to release with
///   `M_attnDispatchFreeError`; may be null.
/// @returns 0 on success. Nonzero for a null argument, an unknown kind, an
///   invalid shape, a needed field left unset, or a shape the kernels do not
///   support.
M_ATTN_DISPATCH_API int32_t M_attnDispatchResolve(
    void *device_ctx, const MAttnDispatchDesc *desc, int64_t batch_size,
    int64_t q_max_seq_len, int64_t max_cache_valid_length, MAttnDispatch *out,
    char **err);

/// Releases a message `M_attnDispatchResolve` returned. Null is a no-op.
M_ATTN_DISPATCH_API void M_attnDispatchFreeError(char *err);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KERNELS_ATTN_DISPATCH_H
