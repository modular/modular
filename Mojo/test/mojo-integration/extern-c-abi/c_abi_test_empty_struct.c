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
// C reference for an empty struct passed by value between two arguments.
// Mojo test: test_empty_struct_arguments.mojo
//
// In GNU C an empty struct has size 0 and takes no register or stack slot, so
// the argument after it is in the register that follows the argument before
// it. Each function gives a * 1000 + b, so a shifted argument shows.

#include <stdint.h>

// The GNU extension is the subject of the test.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-empty-struct"
typedef struct {
} Empty;
#pragma clang diagnostic pop

typedef int64_t (*EmptyBetweenFn)(int32_t, Empty, int32_t);

// Mojo binds this one with @extern.
int64_t c_empty_between_extern(int32_t a, Empty e, int32_t b) {
  (void)e;
  return (int64_t)a * 1000 + b;
}

// Mojo calls this one through external_call.
int64_t c_empty_between(int32_t a, Empty e, int32_t b) {
  (void)e;
  return (int64_t)a * 1000 + b;
}

// Mojo calls the result through an abi("C") function pointer.
EmptyBetweenFn c_get_empty_between(void) { return c_empty_between; }

// C calls a Mojo @export abi("C") function through a pointer.
int64_t c_invoke_empty_between(EmptyBetweenFn fn) {
  Empty e;
  return fn(3, e, 4);
}
