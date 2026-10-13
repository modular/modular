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

#include "Support/DLFcn.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>

namespace {
// dlerror contract: return the message for the most recent failure on the
// calling thread once, then report no error until the next failure.
thread_local char lastError[512];
thread_local bool hasError = false;

void setLastError(const char *what, const char *detail) {
  snprintf(lastError, sizeof(lastError), "%s(%s) failed with error %lu", what,
           detail ? detail : "", GetLastError());
  hasError = true;
}
} // namespace

extern "C" {

void *dlopen(const char *filename, int flags) {
  (void)flags;
  // POSIX: a null filename yields a handle for the calling program itself.
  if (filename == nullptr)
    return GetModuleHandleA(nullptr);
  HMODULE handle = LoadLibraryA(filename);
  if (handle == nullptr)
    setLastError("LoadLibrary", filename);
  return handle;
}

void *dlsym(void *handle, const char *symbol) {
  FARPROC addr = GetProcAddress(static_cast<HMODULE>(handle), symbol);
  if (addr == nullptr)
    setLastError("GetProcAddress", symbol);
  return reinterpret_cast<void *>(addr);
}

int dlclose(void *handle) {
  if (!FreeLibrary(static_cast<HMODULE>(handle))) {
    setLastError("FreeLibrary", nullptr);
    return -1;
  }
  return 0;
}

char *dlerror(void) {
  if (!hasError)
    return nullptr;
  hasError = false;
  return lastError;
}

} // extern "C"

#endif // _WIN32
