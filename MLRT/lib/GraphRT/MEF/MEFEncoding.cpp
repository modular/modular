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

#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "llvm/Support/raw_ostream.h"
using namespace M::MEF;

/// Provides a way to get a MEF attribute kind as a string.
std::string M::MEF::getAsString(AttributeKind kind) {
  // If it's one of these then we can just use a constant string. The containers
  // support multiple different kinds of attribute, so we can't necessarily say
  // what the actual type is.
  if (kind == AttributeKind::kBool)
    return "bool";
  if (kind == AttributeKind::kString)
    return "string";
  if (kind == AttributeKind::kDictionary)
    return "dictionary";
  if (kind == AttributeKind::kList)
    return "list";
  if (kind == AttributeKind::kType)
    return "type";
  if (kind == AttributeKind::kSymbolRef)
    return "symbol ref";

  // Otherwise we should try to print this thing in a way that makes sense.
  std::string out;
  llvm::raw_string_ostream os(out);
  if ((uint8_t)kind & (uint8_t)AttributeKind::mArrayType) {
    os << "array of "
       << getAsString(AttributeKind((uint8_t)kind &
                                    ~(uint8_t)AttributeKind::mArrayType));
    return os.str();
  }

  // Get the prefix with the category.
  switch (getCategory(kind)) {
  case AttrCategory::Float:
    os << "f";
    break;
  case AttrCategory::Signed:
    os << "si";
    break;
  case AttrCategory::Unsigned:
    os << "ui";
    break;
  }

  // And get the width in bits.
  os << getWidthInBits(kind);

  return os.str();
}
