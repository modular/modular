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
// Portable <dlfcn.h>: POSIX systems get the system header; Windows gets a
// minimal dlopen/dlsym/dlclose/dlerror over LoadLibrary/GetProcAddress so
// runtime code that loads drivers and plugins dynamically compiles unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DLFCN_H
#define SUPPORT_DLFCN_H

#ifdef _WIN32

// Mode flags accepted for source compatibility and otherwise ignored: Windows
// has no lazy binding, and DLL exports are always process-global.
#define RTLD_LAZY 0x1
#define RTLD_NOW 0x2
#define RTLD_LOCAL 0x4
#define RTLD_GLOBAL 0x8

extern "C" {
void *dlopen(const char *filename, int flags);
void *dlsym(void *handle, const char *symbol);
int dlclose(void *handle);
char *dlerror(void);
}

#else
#include <dlfcn.h>
#endif // _WIN32

#endif // SUPPORT_DLFCN_H
