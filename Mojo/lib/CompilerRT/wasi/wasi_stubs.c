//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// WASI definitions of KGEN_CompilerRT_* symbols with no host syscall.
// Globals, argv, and the allocator are in kgen_compilerrt_wasi.c and
// wasi_alloc.c.
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

_Bool KGEN_CompilerRT_Initialize(void) { return 1; }
void KGEN_CompilerRT_SetAsanAllocators(void) {}

extern void *KGEN_CompilerRT_GetOrCreateGlobal(const char *name,
                                               size_t name_len,
                                               void *(*init_fn)(void),
                                               void (*destroy_fn)(void *));

void *KGEN_CompilerRT_GetGlobalOrNull(const char *name, size_t name_len) {
  return KGEN_CompilerRT_GetOrCreateGlobal(name, name_len, (void *(*)(void))0,
                                           (void (*)(void *))0);
}

static void *kgen_wasi_insert_value;
static void *kgen_wasi_insert_init_fn(void) { return kgen_wasi_insert_value; }

void KGEN_CompilerRT_InsertGlobal(const char *name, size_t name_len,
                                  void *value) {
  kgen_wasi_insert_value = value;
  (void)KGEN_CompilerRT_GetOrCreateGlobal(
      name, name_len, kgen_wasi_insert_init_fn, (void (*)(void *))0);
  kgen_wasi_insert_value = (void *)0;
}

/* Must match NUM_INDEXED_GLOBALS in std/ffi/__init__.mojo. */
#define KGEN_WASI_NUM_INDEXED_GLOBALS 3
static void *g_indexed_values[KGEN_WASI_NUM_INDEXED_GLOBALS];
static void (*g_indexed_destroys[KGEN_WASI_NUM_INDEXED_GLOBALS])(void *);

void *KGEN_CompilerRT_GetOrCreateGlobalIndexed(size_t index,
                                               void *(*init_fn)(void),
                                               void (*destroy_fn)(void *)) {
  if (index >= KGEN_WASI_NUM_INDEXED_GLOBALS || !init_fn)
    return (void *)0;
  if (g_indexed_values[index])
    return g_indexed_values[index];
  void *value = init_fn();
  if (!value)
    return (void *)0;
  g_indexed_values[index] = value;
  g_indexed_destroys[index] = destroy_fn;
  return value;
}

void KGEN_CompilerRT_DestroyIndexedGlobals(void) {
  for (size_t i = 0; i < KGEN_WASI_NUM_INDEXED_GLOBALS; ++i) {
    void (*destroy_fn)(void *) = g_indexed_destroys[i];
    void *value = g_indexed_values[i];
    g_indexed_destroys[i] = (void (*)(void *))0;
    g_indexed_values[i] = (void *)0;
    if (destroy_fn && value)
      destroy_fn(value);
  }
}

size_t KGEN_CompilerRT_NumPhysicalCores(void) { return 1; }
size_t KGEN_CompilerRT_NumLogicalCores(void) { return 1; }
size_t KGEN_CompilerRT_NumPerformanceCores(void) { return 1; }

/* Mojo passes `depth: Int` as i64 and expects an i64 count. The native
 * prototype is (void **, int) -> int. */
int64_t KGEN_CompilerRT_GetStackTrace(void **strings, int64_t depth) {
  (void)strings;
  (void)depth;
  return 0;
}

/* Chain slot is NULL or this sentinel. AsyncRTAsyncChainRef is one pointer. */
static uint8_t kgen_wasi_chain_complete_sentinel;

void KGEN_CompilerRT_AsyncRT_InitializeChain(void **chain_slot) {
  if (chain_slot)
    *chain_slot = (void *)0;
}

void KGEN_CompilerRT_AsyncRT_DestroyChain(void **chain_slot) {
  if (chain_slot)
    *chain_slot = (void *)0;
}

void KGEN_CompilerRT_AsyncRT_Complete(void **chain_slot) {
  if (chain_slot)
    *chain_slot = &kgen_wasi_chain_complete_sentinel;
}

void KGEN_CompilerRT_AsyncRT_Wait(void **chain_slot) { (void)chain_slot; }

_Bool KGEN_CompilerRT_AsyncRT_Wait_Timeout(void **chain_slot, int64_t ns) {
  (void)chain_slot;
  (void)ns;
  return 1;
}

typedef long kgen_wasi_ssize_t;
void KGEN_CompilerRT_AsyncRT_Execute(void (*resume)(int8_t *), int8_t *hdl,
                                     kgen_wasi_ssize_t desired_worker_id) {
  (void)desired_worker_id;
  if (resume)
    resume(hdl);
}

void KGEN_CompilerRT_AsyncRT_AndThen(void (*resume)(int8_t *),
                                     void **chain_slot, int8_t *hdl) {
  (void)chain_slot;
  if (resume)
    resume(hdl);
}

uint32_t KGEN_CompilerRT_AsyncRT_ParallelismLevel(void) { return 1; }

static uint8_t kgen_wasi_spin_waiter_sentinel;
void *KGEN_CompilerRT_AsyncRT_InitializeSpinWaiter(void) {
  return &kgen_wasi_spin_waiter_sentinel;
}
void KGEN_CompilerRT_AsyncRT_SpinWaiter_Wait(void *waiter) { (void)waiter; }
void KGEN_CompilerRT_AsyncRT_DestroySpinWaiter(void *waiter) { (void)waiter; }

size_t KGEN_CompilerRT_TimeTraceProfilerBeginTask(const char *name_ptr,
                                                  size_t name_len,
                                                  size_t parent_id,
                                                  size_t task_id) {
  (void)name_ptr;
  (void)name_len;
  (void)parent_id;
  (void)task_id;
  return 0;
}

size_t KGEN_CompilerRT_TimeTraceProfilerBeginDetail(const char *name_ptr,
                                                    size_t name_len,
                                                    const char *detail_ptr,
                                                    size_t detail_len,
                                                    size_t parent_id) {
  (void)name_ptr;
  (void)name_len;
  (void)detail_ptr;
  (void)detail_len;
  (void)parent_id;
  return 0;
}

size_t KGEN_CompilerRT_TimeTraceProfilerBegin(const char *name_ptr,
                                              size_t name_len,
                                              size_t parent_id) {
  (void)name_ptr;
  (void)name_len;
  (void)parent_id;
  return 0;
}

void KGEN_CompilerRT_TimeTraceProfilerEnd(size_t id) { (void)id; }

size_t KGEN_CompilerRT_TimeTraceProfilerGetCurrentId(void) { return 0; }

void KGEN_CompilerRT_TimeTraceProfilerSetCurrentId(size_t id) { (void)id; }

size_t KGEN_CompilerRT_GetNextOpId(void) {
  static size_t counter = 0;
  return counter++;
}

size_t KGEN_CompilerRT_TracyIsEnabled(void) { return 0; }

uint64_t KGEN_CompilerRT_TracyZoneBegin(const char *name, size_t name_len,
                                        uint32_t color) {
  (void)name;
  (void)name_len;
  (void)color;
  return 0;
}

void KGEN_CompilerRT_TracyZoneEnd(uint64_t packed_ctx) { (void)packed_ctx; }

/* One thread, one block. Index 0 and dimension 0 would skip every
 * `thread_idx < block_dim` loop. */
int64_t _wasm_get_thread_idx_x(void) { return 0; }
int64_t _wasm_get_thread_idx_y(void) { return 0; }
int64_t _wasm_get_thread_idx_z(void) { return 0; }
int64_t _wasm_get_block_idx_x(void) { return 0; }
int64_t _wasm_get_block_idx_y(void) { return 0; }
int64_t _wasm_get_block_idx_z(void) { return 0; }
int64_t _wasm_get_block_dim_idx_x(void) { return 1; }
int64_t _wasm_get_block_dim_idx_y(void) { return 1; }
int64_t _wasm_get_block_dim_idx_z(void) { return 1; }
int64_t _wasm_get_grid_idx_x(void) { return 1; }
int64_t _wasm_get_grid_idx_y(void) { return 1; }
int64_t _wasm_get_grid_idx_z(void) { return 1; }

#ifdef __cplusplus
}
#endif
