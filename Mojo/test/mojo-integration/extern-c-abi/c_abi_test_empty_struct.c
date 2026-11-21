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
// C reference for empty struct arguments.
// Mojo test: test_empty_struct_arguments.mojo
//
// An empty struct has no ABI representation: it occupies no register and no
// stack slot. Every call site must drop it, or the arguments after it shift
// by one and the callee reads the wrong values.
//
// Each entry point has its own symbol so that the three Mojo call paths do not
// share one LLVM declaration in the test module.

#include <stdint.h>

// An empty struct is a GNU extension in C: it has size zero and no ABI
// representation, which is exactly the case this test covers.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-empty-struct"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
struct Empty {};
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// The argument after `e` is the one that shifts if `e` is not dropped.
int32_t c_empty_struct_extern(int32_t a, struct Empty e, int32_t b) {
  (void)e;
  return a * 1000 + b;
}

int32_t c_empty_struct_external_call(int32_t a, struct Empty e, int32_t b) {
  (void)e;
  return a * 1000 + b;
}

int32_t c_empty_struct_fn_ptr(int32_t a, struct Empty e, int32_t b) {
  (void)e;
  return a * 1000 + b;
}

typedef int32_t (*empty_struct_fn)(int32_t, struct Empty, int32_t);

empty_struct_fn c_get_empty_struct_fn(void) { return c_empty_struct_fn_ptr; }
