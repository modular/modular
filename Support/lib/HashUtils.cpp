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

#include "Support/HashUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

// HashUtils.h already includes xxhash.h (non-inline, with the full
// XXH3_state_t definition). Do not include "xxh3.h": it forces XXH_INLINE_ALL,
// which is incompatible with the dispatcher below.
#include "xxhash.h"

// On x86, route the streaming XXH3 update calls through the runtime CPU
// dispatcher so hashing uses AVX2/AVX-512 when the host supports it. The
// dispatcher redefines XXH3_{64,128}bits_update to *_dispatch variants
// (implemented in xxh_x86dispatch.c, already compiled into @xxhash on x86_64),
// so this include must precede the write_impl definitions below that call
// them. It is confined to this translation unit, so the macro rewrites do not
// leak through HashUtils.h. The reset/digest calls stay on the default path;
// xxHash guarantees bit-identical output across all backends, so mixing them
// is safe.
#if defined(__x86_64__) || defined(_M_X64)
#include "xxh_x86dispatch.h"
#endif

M::raw_xxhash128_stream::raw_xxhash128_stream() : raw_ostream() {
  XXH3_128bits_reset(&State);
}

void M::raw_xxhash128_stream::write_impl(const char *Ptr, size_t Size) {
  XXH3_128bits_update(&State, (void *)Ptr, Size);
}

M::raw_xxhash64_stream::raw_xxhash64_stream() : raw_ostream() {
  XXH3_64bits_reset(&State);
}

void M::raw_xxhash64_stream::write_impl(const char *Ptr, size_t Size) {
  XXH3_64bits_update(&State, (void *)Ptr, Size);
}

std::array<uint8_t, 16> M::raw_xxhash128_stream::hash() {
  flush();

  XXH128_hash_t digest = XXH3_128bits_digest(&State);
  std::array<uint8_t, 16> result;
  memcpy(&result[0], (void *)&digest.low64, 8);
  memcpy(&result[8], (void *)&digest.high64, 8);
  return result;
}

std::string M::raw_xxhash128_stream::hashString() {
  llvm::SmallString<32> output;
  llvm::toHex(hash(), /*LowerCase=*/true, output);
  return std::string(output);
}

uint64_t M::raw_xxhash64_stream::hash() {
  flush();
  return XXH3_64bits_digest(&State);
}
