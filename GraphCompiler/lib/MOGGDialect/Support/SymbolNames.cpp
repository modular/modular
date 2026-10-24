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

#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

static std::string sanitizePart(StringRef part) {
  std::string out;
  for (char c : part) {
    if (isAlnum(c) || c == '_')
      out.push_back(c);
    else if (out.empty() || out.back() != '_')
      out.push_back('_');
  }
  // `shapeof(range)` -> `shapeof_range`, not `shapeof_range_`.
  while (out.size() > 1 && out.back() == '_')
    out.pop_back();
  return out;
}

std::string M::MOGG::joinNameParts(ArrayRef<std::string> parts,
                                   TruncationMarker marker, size_t maxLength) {
  SmallVector<std::string> pieces = map_to_vector(
      parts, [](const std::string &p) { return sanitizePart(p); });

  std::string name;
  size_t kept = 0;
  for (const std::string &piece : pieces) {
    size_t length = name.size() + (name.empty() ? 0 : 1) + piece.size();
    if (kept != 0 && length > maxLength)
      break;
    if (!name.empty())
      name += '_';
    name += piece;
    ++kept;
  }

  if (kept < pieces.size()) {
    if (marker == TruncationMarker::Hash)
      name += "_" + utohexstr(xxh3_64bits(join(pieces, "_")),
                              /*LowerCase=*/true, /*Width=*/16);
    else
      name += "_and" + std::to_string(pieces.size() - kept);
  }

  if (!name.empty() && isDigit(name.front()))
    name.insert(name.begin(), '_');
  return name;
}
