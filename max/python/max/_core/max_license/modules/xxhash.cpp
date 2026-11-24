//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>

#include "xxhash.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;

void M::Graph::Python::bindXXHashModule(nb::module_ m) {
  m.def("xxh3_64_intdigest", [](nb::ndarray<> &arr) {
    return XXH3_64bits_withSeed(arr.data(), arr.nbytes(), 0);
  });

  // Hashes `data` followed by `suffix` as a little-endian u64, without
  // materializing the concatenation.
  //
  // The one-shot entry point above forces the caller to build `data ++ suffix`
  // in one buffer first, and when `data` is a whole media file that copy costs
  // roughly twenty times the hash it feeds: 0.26ms of copy against 0.009ms of
  // hashing for a 409KB JPEG. Streaming the two pieces removes the copy, and
  // XXH3 defines the streamed digest as identical to the one-shot digest over
  // the same byte sequence, so keys computed either way agree -- which matters,
  // because a separate encoder process reproduces these keys.
  m.def(
      "xxh3_64_intdigest_with_u64_suffix",
      [](nb::bytes data, uint64_t suffix) {
        // The vendored xxhash.h only forward-declares XXH3_state_t, so the
        // state cannot live on the stack and has to come from the API (which
        // also gives it the alignment the vector paths need). One small
        // allocation is noise beside the payload copy this avoids.
        const std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)> state{
            XXH3_createState(), &XXH3_freeState};
        if (!state || XXH3_64bits_reset(state.get()) != XXH_OK)
          throw std::runtime_error("failed to initialize the XXH3 state");
        if (data.size() != 0 && XXH3_64bits_update(state.get(), data.c_str(),
                                                   data.size()) != XXH_OK)
          throw std::runtime_error("failed to hash the payload");
        // Serialized explicitly rather than memcpy'd from the integer, so the
        // digest does not depend on the host's byte order.
        std::array<uint8_t, sizeof(uint64_t)> tail{};
        for (size_t i = 0; i < tail.size(); ++i)
          tail[i] = static_cast<uint8_t>((suffix >> (8 * i)) & 0xFF);
        if (XXH3_64bits_update(state.get(), tail.data(), tail.size()) != XXH_OK)
          throw std::runtime_error("failed to hash the suffix");
        return XXH3_64bits_digest(state.get());
      },
      nb::arg("data"), nb::arg("suffix"));
}
