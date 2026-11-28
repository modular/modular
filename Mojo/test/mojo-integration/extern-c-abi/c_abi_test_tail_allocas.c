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
// C reference for caller stack temporaries passed to `tail` calls.
// Mojo test: test_tail_caller_allocas.mojo
//
// A 512-byte struct is passed and returned in memory. The Mojo side moves it
// through a local temporary before calling another Mojo function. If that call
// keeps its `tail` marker, LLVM may assume the callee never touches the
// temporary and delete the copy into it.

#include <stdint.h>
#include <stdio.h>

typedef struct {
  int64_t v[64];
} TailBig;

void c_tail_drive_param(int64_t (*f)(TailBig)) {
  TailBig b;
  for (int i = 0; i < 64; i++)
    b.v[i] = 7;
  printf("sum = %ld\n", (long)f(b));
}

void c_tail_drive_result(TailBig (*f)(int64_t)) {
  TailBig r = f(7);
  printf("v[0] = %ld, v[63] = %ld\n", (long)r.v[0], (long)r.v[63]);
}
