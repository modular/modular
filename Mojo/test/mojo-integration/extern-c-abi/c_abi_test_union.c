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
// C reference for a Mojo __union crossing the C ABI by value.
// Mojo test: test_union_arguments.mojo
//
// A Mojo __union lowers to the struct below: the untagged union storage
// first, then the uint8 tag. The functions take and return it by value, and
// invoke Mojo @export abi("C") functions through pointers, so the round trip
// covers argument and result position in both directions.

#include <stdint.h>

// 16 bytes: { union { int64_t i; double f; } u; uint8_t tag; }
typedef struct {
  union {
    int64_t i;
    double f;
  } u;
  uint8_t tag;
} U;

// Mojo passes a union by value; C reads payload and tag.
int64_t c_take_u(U u) {
  if (u.tag == 0)
    return u.u.i * 10 + u.tag;
  return 500;
}

// C returns a union by value; Mojo reads payload and tag.
U c_give_u(void) {
  U r;
  r.u.f = 2.5;
  r.tag = 1;
  return r;
}

// C invokes a Mojo abi("C") function pointer with a by-value union.
int64_t c_invoke_take(int64_t (*fn)(U)) {
  U r;
  r.u.i = 42;
  r.tag = 0;
  return fn(r);
}

// C invokes a Mojo abi("C") function pointer and reads its by-value union
// result.
double c_invoke_give(U (*fn)(void)) {
  U r = fn();
  if (r.tag == 1)
    return r.u.f;
  return -1.0;
}
