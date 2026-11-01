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
// Readable names for kernel stubs and fusion-info aliases, built from the ops
// a kernel fuses.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_SYMBOLNAMES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_SYMBOLNAMES_H

#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace M::MOGG {

/// The longest name `joinNameParts` builds before it starts dropping parts.
/// Matches what `KGEN::sanitizeSymbolToUnderscores` keeps of a symbol.
inline constexpr size_t kMaxJoinedNameLength = 32;

/// What `joinNameParts` appends when the parts do not all fit.
enum class TruncationMarker {
  /// `_<16 hex digits>`, a hash of every part, so names built from different
  /// parts stay distinct.
  Hash,
  /// `_and<k>`, where `k` is the number of dropped parts.
  DroppedCount,
};

/// Joins `parts` with `_` into an identifier: characters outside
/// `[A-Za-z0-9_]` become `_`, and a leading digit gets a `_` prefix. Parts are
/// added whole until the next one would make the name longer than
/// `maxLength`, so no part is ever cut; the first part is always kept. When
/// parts are dropped, `marker` says what is appended.
std::string joinNameParts(llvm::ArrayRef<std::string> parts,
                          TruncationMarker marker,
                          size_t maxLength = kMaxJoinedNameLength);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_SYMBOLNAMES_H
