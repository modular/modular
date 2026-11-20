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

#include <cstdint>
#include <cstdlib>

/// Stores the sum of the `count` values at `*values` in `*sum`, and returns
/// `count` plus `INTERPRETER_LIB_OFFSET`, which stands in for a dependency the
/// compiler does not track.
extern "C" __attribute__((visibility("default"))) int64_t
sum_indirect(const int64_t *const *values, int64_t count, int64_t *sum) {
  int64_t total = 0;
  for (int64_t i = 0; i < count; ++i)
    total += (*values)[i];
  *sum = total;
  const char *offset = std::getenv("INTERPRETER_LIB_OFFSET");
  return count + (offset ? std::atoll(offset) : 0);
}
