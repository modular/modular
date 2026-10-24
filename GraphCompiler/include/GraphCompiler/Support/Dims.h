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
//
// Functions for dim manipulation.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_DIMS_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_DIMS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace M {

inline int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return (axis < 0) ? rank + axis : axis;
}

/// Clamp axis to the range [0, rank].
inline int64_t clampAxis(int64_t axis, int64_t rank) {
  int64_t normalizedAxis = normalizeAxis(axis, rank);
  return std::max(std::min(normalizedAxis, rank), (int64_t)0);
}

/// Normalize negative dims in `dims` to their positive equivalent.
inline void normalizeNegativeDims(llvm::MutableArrayRef<int64_t> dims,
                                  size_t rank) {
  for (auto [i, dim] : llvm::enumerate(dims))
    dims[i] = normalizeAxis(dim, rank);
}

} // namespace M

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_DIMS_H
