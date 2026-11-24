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
// The C ABI between the HAL DeviceContext shim (HALDeviceContext.cpp) and its
// Mojo half (hal_device_context_exports.mojo). This is the only declaration of
// the MojoHAL_* exports; //MLRT/unittests/Driver:test_mojohal_exports_abi
// checks each one against its Mojo definition.
//
// An entry that returns `const char *` returns null on success, or an error
// message the caller frees with `MojoHAL_strfree`.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HAL_MOJOHALEXPORTS_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HAL_MOJOHALEXPORTS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Heap-boxed Mojo values, created and released through the entries below.
typedef struct MojoHALContextBox MojoHALContextBox;
typedef struct MojoHALBufferBox MojoHALBufferBox;
typedef struct MojoHALScopeBox MojoHALScopeBox;
typedef struct MojoHALCompletionFlagBox MojoHALCompletionFlagBox;
typedef struct MojoHALStreamBox MojoHALStreamBox;
typedef struct MojoHALEventBox MojoHALEventBox;

void MojoHAL_strfree(const char *s);

// ===-- Devices ---------------------------------------------------------=== //

int32_t MojoHAL_Device_numberOfDevices(const char *api);
const char *MojoHAL_Device_getApiAndArch(char **out_api, char **out_arch,
                                         int32_t device_id);
const char *MojoHAL_DeviceContext_enableAllPeerAccess(void);
const char *MojoHAL_DeviceContext_allPeerAccessEnabled(bool *out_enabled);
const char *MojoHAL_DeviceContext_enablePeerAccess(MojoHALContextBox *ctx,
                                                   MojoHALContextBox *peer);

// ===-- Contexts --------------------------------------------------------=== //

const char *MojoHAL_DeviceContext_create(MojoHALContextBox **out_ctx,
                                         const char *api, int32_t device_id);
void MojoHAL_DeviceContext_copy(MojoHALContextBox **out_ctx,
                                MojoHALContextBox *ctx);
void MojoHAL_DeviceContext_release(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_synchronize(MojoHALContextBox *ctx);

// ===-- Buffers ---------------------------------------------------------=== //

const char *MojoHAL_DeviceContext_createBuffer(MojoHALBufferBox **out_buf,
                                               void **out_ptr,
                                               MojoHALContextBox *ctx,
                                               uint64_t byte_size);
const char *MojoHAL_DeviceContext_createBufferOwning(MojoHALBufferBox **out_buf,
                                                     MojoHALContextBox *ctx,
                                                     void *ptr,
                                                     uint64_t byte_size,
                                                     bool owning);
void MojoHAL_DeviceBuffer_release(MojoHALBufferBox *buf);
const char *MojoHAL_DeviceBuffer_relinquishOwnership(MojoHALBufferBox *buf);
const char *MojoHAL_DeviceContext_createHostBuffer(MojoHALBufferBox **out_buf,
                                                   void **out_ptr,
                                                   MojoHALContextBox *ctx,
                                                   uint64_t byte_size);
void MojoHAL_HostBuffer_release(MojoHALBufferBox *buf);
const char *MojoHAL_HostBuffer_createSubBuffer(MojoHALBufferBox **out_buf,
                                               MojoHALBufferBox *parent,
                                               uint64_t byte_offset,
                                               uint64_t byte_size);
const char *MojoHAL_DeviceBuffer_createSubBuffer(MojoHALBufferBox **out_buf,
                                                 void **out_ptr,
                                                 MojoHALBufferBox *parent,
                                                 uint64_t byte_offset,
                                                 uint64_t byte_size);
const char *MojoHAL_DeviceContext_memmapFile(MojoHALBufferBox **out_buf,
                                             void **out_ptr, uint64_t *out_size,
                                             MojoHALContextBox *ctx,
                                             const char *path);
void MojoHAL_MmapBuffer_release(MojoHALBufferBox *buf);

// ===-- Copies / memset -------------------------------------------------=== //

const char *MojoHAL_DeviceContext_enqueueCopy(MojoHALContextBox *ctx, void *dst,
                                              const void *src,
                                              uint64_t byte_size);
const char *MojoHAL_DeviceContext_enqueueCopyToDevice(MojoHALContextBox *ctx,
                                                      void *dst,
                                                      const void *src,
                                                      uint64_t byte_size);
const char *MojoHAL_DeviceContext_enqueueCopyFromDevice(MojoHALContextBox *ctx,
                                                        void *dst,
                                                        const void *src,
                                                        uint64_t byte_size);
const char *MojoHAL_DeviceContext_enqueueSetMemory(MojoHALContextBox *ctx,
                                                   void *dst,
                                                   uint64_t byte_size,
                                                   uint64_t value,
                                                   uint64_t value_size);

// ===-- Device info -----------------------------------------------------=== //

const char *MojoHAL_DeviceContext_getArchName(char **out_name,
                                              MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_isHostUnified(bool *out_unified,
                                                MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_maxAllocationSize(bool *out_has_limit,
                                                    uint64_t *out_size,
                                                    MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_maxContextsPerDevice(bool *out_has_limit,
                                                       int64_t *out_limit,
                                                       MojoHALContextBox *ctx);
bool MojoHAL_DeviceContext_isCompatible(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getName(char **out_name,
                                          MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getApi(char **out_api,
                                         MojoHALContextBox *ctx);
int64_t MojoHAL_DeviceContext_id(MojoHALContextBox *ctx);
bool MojoHAL_DeviceContext_canAccess(MojoHALContextBox *ctx,
                                     MojoHALContextBox *peer);
const char *MojoHAL_DeviceContext_getMemoryInfo(uint64_t *out_free,
                                                uint64_t *out_total,
                                                MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getAttribute(int32_t *out_value,
                                               MojoHALContextBox *ctx,
                                               int32_t attr);

// ===-- Streams / events / thread binding -------------------------------=== //

const char *MojoHAL_DeviceContext_reassignWait(MojoHALContextBox *dst_ctx,
                                               MojoHALContextBox *src_ctx);
bool MojoHAL_DeviceContext_sameStream(MojoHALContextBox *a,
                                      MojoHALContextBox *b);
const char *MojoHAL_DeviceContext_stream(MojoHALStreamBox **out_stream,
                                         MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_createStream(MojoHALStreamBox **out_stream,
                                               MojoHALContextBox *ctx,
                                               int32_t priority);
const char *MojoHAL_DeviceContext_withStream(MojoHALContextBox **out_ctx,
                                             MojoHALContextBox *ctx,
                                             MojoHALStreamBox *stream);
void MojoHAL_DeviceStream_release(MojoHALStreamBox *stream);
const char *MojoHAL_DeviceStream_synchronize(MojoHALStreamBox *stream);
const char *MojoHAL_DeviceStream_waitForEvent(MojoHALStreamBox *stream,
                                              MojoHALEventBox *event);
const char *MojoHAL_DeviceStream_nativeHandle(void **out_handle,
                                              MojoHALStreamBox *stream);
const char *MojoHAL_DeviceEvent_isReady(bool *out_ready,
                                        MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_eventCreate(MojoHALEventBox **out_event,
                                              MojoHALContextBox *ctx,
                                              bool record);
void MojoHAL_DeviceEvent_release(MojoHALEventBox *event);
const char *MojoHAL_DeviceEvent_synchronize(MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_enqueueEventRecord(MojoHALContextBox *ctx,
                                                     MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_enqueueWaitForEvent(MojoHALContextBox *ctx,
                                                      MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_bindToCurrentThread(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_pushContext(MojoHALScopeBox **out_scope,
                                              MojoHALContextBox *ctx);
void MojoHAL_DeviceContextScope_pop(MojoHALScopeBox *scope);
// Returns false when the backend has no native stream handle.
bool MojoHAL_DeviceContext_getDLPackStream(intptr_t *out_stream,
                                           MojoHALContextBox *ctx);

// ===-- Completion flags / host-value waits -----------------------------=== //

const char *MojoHAL_DeviceContext_allocateCompletionFlag(
    MojoHALCompletionFlagBox **out_flag, MojoHALContextBox *ctx);
bool MojoHAL_DeviceContext_supportsCompletionFlags(MojoHALContextBox *ctx);
void MojoHAL_CompletionFlag_release(MojoHALCompletionFlagBox *flag);
const char *MojoHAL_CompletionFlag_signal(MojoHALCompletionFlagBox *flag,
                                          uint64_t value);
const char *MojoHAL_CompletionFlag_load(MojoHALCompletionFlagBox *flag,
                                        uint64_t *out_value);
const char *MojoHAL_CompletionFlag_reset(MojoHALCompletionFlagBox *flag);
const char *MojoHAL_DeviceStream_launchHostFunc(MojoHALStreamBox *stream,
                                                void (*func)(void *),
                                                void *user_data);
const char *MojoHAL_DeviceStream_waitOnFlag(MojoHALStreamBox *stream,
                                            MojoHALCompletionFlagBox *flag,
                                            uint64_t value);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HAL_MOJOHALEXPORTS_H
