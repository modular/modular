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

#include "MLRT/GraphRT/MEF/MEFAttr.h"

using namespace M;
using namespace Detail;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Given a pointer to the start of a MEF array or string attribute, return the
/// length of array or string by decoding the prefix encoded VBR.
size_t M::decodeMEFArrayLength(const uint8_t *dataPtr) {
  // The attribute's kind is between the VBR and the actual value, so we have to
  // make sure to subtract that offset (hence the -2).
  const uint8_t *lenPtr = dataPtr - 2;
  size_t size = 0;
  bool hasMoreBytes;
  do {
    hasMoreBytes = (lenPtr[0] & 0x80) != 0;
    size <<= 7;
    size |= lenPtr[0] & 0x7F;
    --lenPtr;
  } while (hasMoreBytes);
  return size;
}

//===----------------------------------------------------------------------===//
// MEFListAttr implementation
//===----------------------------------------------------------------------===//

MEFListAttr::Iterator &MEFListAttr::Iterator::operator+=(
    MEFListAttr::Iterator::Base::difference_type n) {
  const uint8_t *tempPtr = ptr;
  for (auto i = 0; i < n; ++i) {
    // We have to increment our current pointer past the VBR for each element.
    (void)MEF::parseVBR(tempPtr);
    ++iterIdx;
  }
  ptr = const_cast<uint8_t *>(tempPtr);
  return *this;
}

MEFDynamicAttr MEFListAttr::Iterator::operator*() const {
  const uint8_t *workPtr = ptr;
  return MEFDynamicAttr(basePtr - MEF::parseVBR(workPtr), attrSection);
}

//===----------------------------------------------------------------------===//
// MEFDictionaryAttr::Iterator
//===----------------------------------------------------------------------===//

MEFDictionaryAttr::Iterator &MEFDictionaryAttr::Iterator::operator+=(
    MEFAttr<Dictionary>::Iterator::Base::difference_type n) {
  const uint8_t *tempPtr = ptr;
  for (auto i = 0; i < n; ++i) {
    // We have to increment our current pointer past the VBRs and the inline
    // name.
    (void)MEF::parseVBR(tempPtr);
    (void)MEF::parseVBR(tempPtr);
    ++iterIdx;
  }
  ptr = const_cast<uint8_t *>(tempPtr);
  return *this;
}

MEFDictionaryAttr::Element MEFDictionaryAttr::Iterator::operator*() const {
  const uint8_t *workptr = ptr;
  size_t nameOffset = MEF::parseVBR(workptr);
  size_t dataOffset = MEF::parseVBR(workptr);
  return Element{basePtr, nameOffset, dataOffset, attrSection};
}

//===----------------------------------------------------------------------===//
// MEFDictionaryAttr implementation
//===----------------------------------------------------------------------===//

std::optional<MEFDynamicAttr> MEFDictionaryAttr::find(StringRef name) const {
  auto found = llvm::find_if(
      *this, [&](const Element &elt) { return elt.getName() == name; });

  if (found != end())
    return (*found).getValue();

  return std::nullopt;
}
