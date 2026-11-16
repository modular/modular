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
// C host for the interop/interoperability.mdx example. It makes the same
// call as the page's main.c, but loads the library with dlopen() because
// Bazel builds this binary before the lit test builds the Mojo library.

#include <dlfcn.h>
#include <stdio.h>

typedef long (*add_t)(long, long);

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <shared-lib-path>\n", argv[0]);
    return 2;
  }
  void *lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }
  add_t add = (add_t)dlsym(lib, "add");
  if (!add) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 2;
  }
  printf("sum = %ld\n", add(5, 103));
  return 0;
}
