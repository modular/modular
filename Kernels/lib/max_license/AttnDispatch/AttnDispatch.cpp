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

#include "Kernels/AttnDispatch.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

// The Mojo entry reads and writes both structs by byte offset.
static_assert(offsetof(MAttnDispatchDesc, struct_size) == 0);
static_assert(offsetof(MAttnDispatchDesc, kind) == 8);
static_assert(offsetof(MAttnDispatchDesc, num_q_heads) == 16);
static_assert(offsetof(MAttnDispatchDesc, num_kv_heads) == 24);
static_assert(offsetof(MAttnDispatchDesc, head_dim) == 32);
static_assert(offsetof(MAttnDispatchDesc, page_size) == 40);
static_assert(offsetof(MAttnDispatchDesc, window_size) == 48);
static_assert(offsetof(MAttnDispatchDesc, sparse_indices_stride) == 56);
static_assert(offsetof(MAttnDispatchDesc, kv_dtype) == 64);
static_assert(offsetof(MAttnDispatchDesc, kv_form) == 68);
static_assert(offsetof(MAttnDispatchDesc, flags) == 72);
static_assert(offsetof(MAttnDispatch, key) == 0);
static_assert(offsetof(MAttnDispatch, n_key) == 64);
static_assert(offsetof(MAttnDispatch, metadata) == 72);
static_assert(offsetof(MAttnDispatch, n_metadata) == 136);

// Defined in attn_dispatch_c_api.mojo.
extern "C" int32_t
attn_dispatch_resolve_c(void *deviceCtx, const void *desc, int64_t batchSize,
                        int64_t qMaxSeqLen, int64_t maxCacheValidLength,
                        void *out, char *errorBuffer, int64_t errorCapacity);

static int32_t fail(char **err, const char *message) {
  if (err)
    *err = strdup(message);
  return 1;
}

int32_t M_attnDispatchResolve(void *device_ctx, const MAttnDispatchDesc *desc,
                              int64_t batch_size, int64_t q_max_seq_len,
                              int64_t max_cache_valid_length,
                              MAttnDispatch *out, char **err) {
  if (err)
    *err = nullptr;
  if (!device_ctx)
    return fail(err, "M_attnDispatchResolve: device_ctx is null");
  if (!desc)
    return fail(err, "M_attnDispatchResolve: desc is null");
  if (!out)
    return fail(err, "M_attnDispatchResolve: out is null");

  char message[512] = {};
  if (attn_dispatch_resolve_c(device_ctx, desc, batch_size, q_max_seq_len,
                              max_cache_valid_length, out, message,
                              sizeof(message)) == 0)
    return 0;
  return fail(err, message);
}

void M_attnDispatchFreeError(char *err) { free(err); }
