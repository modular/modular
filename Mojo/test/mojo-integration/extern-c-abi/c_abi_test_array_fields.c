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
// C ABI reference for array-field struct tests
// Mojo test: test_struct_arguments_array_fields.mojo
//
// Array fields flatten for ABI classification, exactly as the equivalent
// flat fields would: a struct { float v[3]; } is an HFA on AAPCS64 (s0-s2)
// and SSE-classified on x86-64. A lowering that does not look through array
// types classifies them as integer composites and corrupts the values.

#include <stdint.h>

typedef struct float_arr3_t {
  float v[3];
} float_arr3_t;

typedef struct float_arr5_t {
  float v[5];
} float_arr5_t;

typedef struct double_arr3_t {
  double v[3];
} double_arr3_t;

typedef struct int_arr2_t {
  int32_t v[2];
} int_arr2_t;

typedef struct mixed_arr_t {
  float v[2];
  int32_t w;
} mixed_arr_t;

// Each function returns the number of fields that do not hold their expected
// value, so 0 means the struct arrived intact.

int check_float_arr3(float_arr3_t s) {
  int status = 0;
  if (s.v[0] != 11.0f)
    status += 1;
  if (s.v[1] != 22.0f)
    status += 1;
  if (s.v[2] != 33.0f)
    status += 1;
  return status;
}

int check_float_arr5(float_arr5_t s) {
  int status = 0;
  for (int i = 0; i < 5; ++i)
    if (s.v[i] != (float)(i + 1))
      status += 1;
  return status;
}

int check_double_arr3(double_arr3_t s) {
  int status = 0;
  if (s.v[0] != 1.5)
    status += 1;
  if (s.v[1] != 2.5)
    status += 1;
  if (s.v[2] != 3.5)
    status += 1;
  return status;
}

int check_int_arr2(int_arr2_t s) {
  int status = 0;
  if (s.v[0] != 7)
    status += 1;
  if (s.v[1] != 8)
    status += 1;
  return status;
}

int check_mixed_arr(mixed_arr_t s) {
  int status = 0;
  if (s.v[0] != 1.0f)
    status += 1;
  if (s.v[1] != 2.0f)
    status += 1;
  if (s.w != 42)
    status += 1;
  return status;
}
