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
// WASI backend for the KGEN_CompilerRT_* symbols a Mojo program links.
// libc and compiler-rt builtins come from the wasi-libc sysroot.
//
//===----------------------------------------------------------------------===//

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wasi/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* AsyncRTRuntimeRef is one pointer, returned by value. */
static uint8_t kCpuDeviceSentinel;

void *KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice(void) {
  return &kCpuDeviceSentinel;
}

void *KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice(void) {
  return &kCpuDeviceSentinel;
}

void KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice(void *rt) { (void)rt; }

/* StringRef arrives as (name, name_len). Fixed-capacity table: a 65th
 * distinct name returns NULL (native is unbounded). */
#define KGEN_WASI_MAX_GLOBALS 64

typedef struct {
  const char *name_ptr;
  size_t name_len;
  void *value;
  void (*destroy)(void *);
} KgenGlobalEntry;

static KgenGlobalEntry g_globals[KGEN_WASI_MAX_GLOBALS];
static size_t g_globals_count = 0;

/* Hashed index over g_globals, mirroring the native GlobalTable: FNV-1a
 * with linear probing. Stores slot+1 so 0 marks an empty bucket. With at
 * most 64 globals in 128 buckets the load factor stays <= 50%, so a probe
 * always terminates (every probe chain meets an empty bucket). */
#define KGEN_WASI_GLOBAL_INDEX_SIZE 128

/* Probe termination needs spare buckets; the slot+1 encoding must fit. */
_Static_assert(KGEN_WASI_MAX_GLOBALS < KGEN_WASI_GLOBAL_INDEX_SIZE,
               "globals index must keep spare buckets for probe termination");
_Static_assert(KGEN_WASI_MAX_GLOBALS <= UINT8_MAX - 1,
               "slot+1 must fit the uint8_t index buckets");

static uint8_t g_global_index[KGEN_WASI_GLOBAL_INDEX_SIZE];

static size_t kgen_hash_name(const char *s, size_t len) {
  size_t h = (size_t)1469598103934665603ull;
  for (size_t i = 0; i < len; ++i) {
    h ^= (size_t)(unsigned char)s[i];
    h *= (size_t)1099511628211ull;
  }
  return h;
}

static int kgen_streq(const char *a, size_t alen, const char *b, size_t blen) {
  if (alen != blen)
    return 0;
  return memcmp(a, b, alen) == 0;
}

static KgenGlobalEntry *kgen_find_global(const char *name, size_t name_len) {
  size_t mask = KGEN_WASI_GLOBAL_INDEX_SIZE - 1;
  size_t slot = kgen_hash_name(name, name_len) & mask;
  for (;;) {
    uint8_t v = g_global_index[slot];
    if (!v)
      return (KgenGlobalEntry *)0;
    KgenGlobalEntry *e = &g_globals[v - 1];
    if (kgen_streq(e->name_ptr, e->name_len, name, name_len))
      return e;
    slot = (slot + 1) & mask;
  }
}

static void kgen_index_global(size_t idx, const char *name, size_t name_len) {
  size_t mask = KGEN_WASI_GLOBAL_INDEX_SIZE - 1;
  size_t slot = kgen_hash_name(name, name_len) & mask;
  while (g_global_index[slot])
    slot = (slot + 1) & mask;
  g_global_index[slot] = (uint8_t)(idx + 1);
}

void *KGEN_CompilerRT_GetOrCreateGlobal(const char *name, size_t name_len,
                                        void *(*init_fn)(void),
                                        void (*destroy_fn)(void *)) {
  KgenGlobalEntry *found = kgen_find_global(name, name_len);
  if (found)
    return found->value;
  /* A null init is a lookup. Do not insert it; a later create would skip init.
   */
  if (!init_fn || g_globals_count >= KGEN_WASI_MAX_GLOBALS)
    return (void *)0;
  void *value = init_fn();
  if (!value)
    return (void *)0;
  char *owned = (char *)malloc(name_len + 1);
  if (!owned) {
    if (destroy_fn)
      destroy_fn(value);
    return (void *)0;
  }
  memcpy(owned, name, name_len);
  owned[name_len] = '\0';
  size_t idx = g_globals_count++;
  KgenGlobalEntry *e = &g_globals[idx];
  e->name_ptr = owned;
  e->name_len = name_len;
  e->value = value;
  e->destroy = destroy_fn;
  kgen_index_global(idx, owned, name_len);
  return value;
}

extern void KGEN_CompilerRT_DestroyIndexedGlobals(void);

void KGEN_CompilerRT_DestroyGlobals(void) {
  memset(g_global_index, 0, sizeof(g_global_index));
  while (g_globals_count > 0) {
    KgenGlobalEntry *e = &g_globals[--g_globals_count];
    if (e->destroy && e->value)
      e->destroy(e->value);
    free((void *)e->name_ptr);
    e->name_ptr = (const char *)0;
    e->name_len = 0;
    e->value = (void *)0;
    e->destroy = (void (*)(void *))0;
  }
  KGEN_CompilerRT_DestroyIndexedGlobals();
}

void KGEN_CompilerRT_PrintStackTraceOnFault(void) {}

static int32_t g_argc;
static char **g_argv;

static void *g_argv_stringrefs;
static size_t g_argv_stringrefs_count;

void KGEN_CompilerRT_SetArgV(int32_t argc, char **argv) {
  g_argc = argc;
  g_argv = argv;
  g_argv_stringrefs = (void *)0;
  g_argv_stringrefs_count = 0;
}

/* Out-param is { StringRef *args; size_t size }. Cached; never freed. */
typedef struct {
  const char *data;
  size_t length;
} kgen_wasi_stringref_t;

typedef struct {
  kgen_wasi_stringref_t *args;
  size_t size;
} kgen_wasi_rawlist_t;

extern void *KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size);

static size_t kgen_strlen(const char *s) {
  size_t n = 0;
  while (s[n] != '\0')
    ++n;
  return n;
}

/* Publish a one-element argv holding the empty string when no arguments
 * exist, so callers always see at least argv[0]. */
static void kgen_publish_empty_argv(void) {
  kgen_wasi_stringref_t *arr =
      (kgen_wasi_stringref_t *)KGEN_CompilerRT_AlignedAlloc(
          (ssize_t)sizeof(void *), (ssize_t)sizeof(kgen_wasi_stringref_t));
  if (!arr)
    return;
  arr[0].data = "";
  arr[0].length = 0;
  g_argv_stringrefs = arr;
  g_argv_stringrefs_count = 1;
}

static void kgen_build_argv_from_setargv(void) {
  size_t n = (g_argc > 0) ? (size_t)g_argc : 0;
  if (n == 0) {
    kgen_publish_empty_argv();
    return;
  }
  kgen_wasi_stringref_t *arr =
      (kgen_wasi_stringref_t *)KGEN_CompilerRT_AlignedAlloc(
          (ssize_t)sizeof(void *),
          (ssize_t)(n * sizeof(kgen_wasi_stringref_t)));
  if (!arr)
    return;
  for (size_t i = 0; i < n; ++i) {
    const char *s = g_argv[i] ? g_argv[i] : "";
    arr[i].data = s;
    arr[i].length = kgen_strlen(s);
  }
  g_argv_stringrefs = arr;
  g_argv_stringrefs_count = n;
}

static void kgen_build_argv_from_wasi(void) {
  __wasi_size_t argc = 0;
  __wasi_size_t buf_size = 0;
  if (__wasi_args_sizes_get(&argc, &buf_size) != 0 || argc == 0) {
    kgen_publish_empty_argv();
    return;
  }

  uint8_t **argv_ptrs = (uint8_t **)KGEN_CompilerRT_AlignedAlloc(
      (ssize_t)sizeof(void *), (ssize_t)(argc * sizeof(uint8_t *)));
  uint8_t *argv_buf =
      (uint8_t *)KGEN_CompilerRT_AlignedAlloc((ssize_t)1, (ssize_t)buf_size);
  kgen_wasi_stringref_t *arr =
      (kgen_wasi_stringref_t *)KGEN_CompilerRT_AlignedAlloc(
          (ssize_t)sizeof(void *),
          (ssize_t)(argc * sizeof(kgen_wasi_stringref_t)));
  if (!argv_ptrs || !argv_buf || !arr) {
    free(argv_ptrs);
    free(argv_buf);
    free(arr);
    return;
  }

  if (__wasi_args_get(argv_ptrs, argv_buf) != 0) {
    free(argv_ptrs);
    free(argv_buf);
    free(arr);
    return;
  }

  for (__wasi_size_t i = 0; i < argc; ++i) {
    const char *s = (const char *)argv_ptrs[i];
    arr[i].data = s;
    arr[i].length = kgen_strlen(s);
  }
  free(argv_ptrs); /* scratch: the stringrefs point into argv_buf */
  g_argv_stringrefs = arr;
  g_argv_stringrefs_count = (size_t)argc;
}

void KGEN_CompilerRT_GetArgV(kgen_wasi_rawlist_t *result) {
  if (!result)
    return;
  if (g_argv_stringrefs == (void *)0) {
    if (g_argv != (void *)0) {
      kgen_build_argv_from_setargv();
    } else {
      kgen_build_argv_from_wasi();
    }
  }
  result->args = g_argv_stringrefs;
  result->size = g_argv_stringrefs_count;
}

void KGEN_CompilerRT_AlignedFree(void *ptr) { free(ptr); }

/* Cached std-stream FILEs. Without dup(2) the stdlib cannot fclose without
 * closing the fd, so print paths reuse one FILE per std fd instead of
 * leaking one per call. The cache is keyed by fd only: the first mode
 * requested for a std fd wins, and later calls with a different mode get
 * the cached FILE (the three std fds each have one fixed mode in practice).
 * Other fds fall through to plain fdopen. */
static FILE *g_std_stream_files[3];

FILE *KGEN_CompilerRT_fdopen_std(int32_t fd, const char *mode) {
  if (fd >= 0 && fd < 3) {
    if (!g_std_stream_files[fd]) {
      FILE *f = fdopen(fd, mode);
      if (f)
        g_std_stream_files[fd] = f;
    }
    return g_std_stream_files[fd];
  }
  return fdopen(fd, mode);
}

int KGEN_CompilerRT_fprintf(void *stream, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  int result = vfprintf((FILE *)stream, format, ap);
  va_end(ap);
  return result;
}

#ifdef __cplusplus
}
#endif
