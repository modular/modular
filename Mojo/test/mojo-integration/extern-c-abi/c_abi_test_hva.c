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
// C ABI reference for HVA (Homogeneous Vector Aggregate) tests
// Mojo test: test_struct_arguments_hva.mojo
//
// AAPCS64 tests short vectors by size (64 or 128 bits), not element type:
// a struct of two 128-bit integer vectors is an HVA in v0-v1, and so is a
// mix of float32x4 and float64x2. A lowering that requires float elements
// passes such structs by pointer and returns them through x8, which a C
// caller never matches.

#include <stdint.h>

typedef int32_t v4si __attribute__((vector_size(16)));
typedef int32_t v2si __attribute__((vector_size(8)));
typedef float v4sf __attribute__((vector_size(16)));
typedef float v2sf __attribute__((vector_size(8)));
typedef double v2df __attribute__((vector_size(16)));

// Two 128-bit integer vectors: an HVA in v0-v1.
typedef struct hva_i32x4x2_t {
  v4si a;
  v4si b;
} hva_i32x4x2_t;

// A 128-bit float32x4 and a 128-bit float64x2: same size, so still an HVA.
typedef struct hva_mixed128_t {
  v4sf a;
  v2df b;
} hva_mixed128_t;

// A 64-bit integer vector and a 64-bit float vector: same size, an HVA.
typedef struct hva_mixed64_t {
  v2si a;
  v2sf b;
} hva_mixed64_t;

// A 128-bit and a 64-bit vector: different sizes, not homogeneous.
typedef struct not_hva_t {
  v4si a;
  v2si b;
} not_hva_t;

hva_i32x4x2_t c_func_hva_i32x4x2(hva_i32x4x2_t s) {
  for (int i = 0; i < 4; ++i) {
    s.a[i] += 1;
    s.b[i] += 1;
  }
  return s;
}

hva_mixed128_t c_func_hva_mixed128(hva_mixed128_t s) {
  for (int i = 0; i < 4; ++i)
    s.a[i] += 1.0f;
  for (int i = 0; i < 2; ++i)
    s.b[i] += 1.0;
  return s;
}

hva_mixed64_t c_func_hva_mixed64(hva_mixed64_t s) {
  s.a[0] += 1;
  s.a[1] += 1;
  s.b[0] += 1.0f;
  s.b[1] += 1.0f;
  return s;
}

not_hva_t c_func_not_hva(not_hva_t s) {
  for (int i = 0; i < 4; ++i)
    s.a[i] += 1;
  s.b[0] += 1;
  s.b[1] += 1;
  return s;
}
