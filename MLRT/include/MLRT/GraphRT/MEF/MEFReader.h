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
// MEFReader is decoding logic shared between MEF2MLIR and the MEFExecutor.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEF_MEFREADER_H
#define MLRT_GRAPHRT_MEF_MEFREADER_H

#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "Support/Error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

namespace M {

/// This helper class implements low level binary parsing logic for a byte
/// range.  The general protocol for failable parsing routines is that they
/// return false and fill in their operand on success.  On failure, they emit an
/// error to the `errorFunction` and return true.
class MEFReader {
public:
  explicit MEFReader(ArrayRef<uint8_t> contents,
                     llvm::unique_function<void(Error) const> errorFunction)
      : contents(contents), errorFunction(std::move(errorFunction)) {}

  // Return true if we've read the entire file.
  bool empty() const { return contents.empty(); }

  ArrayRef<uint8_t> getContents() const { return contents; }
  const uint8_t *data() const { return contents.data(); }

  /// Reset the contents that this reader will parse from to the specified
  /// range.
  void setContents(ArrayRef<uint8_t> data) { contents = data; }

  bool emitError(Error message) const {
    errorFunction(std::move(message));
    return true;
  }

  /// Parse a single byte from the stream.
  bool parseByte(uint8_t &value) {
    if (contents.empty())
      return emitError("ran out of bytes");
    value = contents.front();
    contents = contents.drop_front();
    return false;
  }

  /// Parse the byte ending at the specified offset.
  bool parseByteAt(size_t offset, uint8_t &value) const {
    if (contents.empty())
      return emitError("ran out of bytes");
    if (offset >= contents.size())
      return emitError("offset out of range");
    value = contents[offset];
    return false;
  }

  /// Parse two bytes in little endian format from the stream.
  bool parseInt2(uint16_t &value) {
    uint8_t a = 0, b = 0;
    if (parseByte(a) || parseByte(b))
      return true;
    value = uint16_t(a) | (uint16_t(b) << 8);
    return false;
  }

  /// Parse four bytes in little endian format from the stream.
  bool parseInt4(uint32_t &value) {
    uint16_t a = 0, b = 0;
    if (parseInt2(a) || parseInt2(b))
      return true;
    value = uint32_t(a) | (uint32_t(b) << 16);
    return false;
  }

  /// Parse N int4 values as an array.
  bool parseInt4Array(size_t numElements, SmallVectorImpl<uint32_t> &result) {
    while (numElements--) {
      result.push_back(0);
      if (parseInt4(result.back()))
        return true;
    }
    return false;
  }

  /// Parse eight bytes in little endian format from the stream.
  bool parseInt8(uint64_t &value) {
    uint32_t a = 0, b = 0;
    if (parseInt4(a) || parseInt4(b))
      return true;
    value = uint64_t(a) | (uint64_t(b) << 32);
    return false;
  }

  /// Parse N int8 values as an array.
  bool parseInt8Array(size_t numElements, SmallVectorImpl<uint64_t> &result) {
    while (numElements--) {
      result.push_back(0);
      if (parseInt8(result.back()))
        return true;
    }
    return false;
  }

  /// Parse a VBR encoded integer from the byte stream.
  bool parseVBR(size_t &result) {
    uint8_t nextByte;
    if (parseByte(nextByte))
      return true;

    result = nextByte & 127;
    while ((nextByte & 128) != 0) {
      if (parseByte(nextByte))
        return true;

      result = (result << 7) | size_t(nextByte & 127);
    }
    return false;
  }

  /// A SignedVBR is emitted with the sign bit as the low bit and negates
  /// negative numbers so we can emit things like "-2" into a single byte.
  bool parseSignedVBR(ssize_t &result) {
    size_t encoded;
    if (parseVBR(encoded))
      return true;
    result = encoded >> 1;
    if (encoded & 1)
      result = -result;
    return false;
  }

  /// Parse an inverted VBR that ends at the specified offset.
  bool parseInvertedVBRAt(size_t offset, size_t &result) const {
    result = 0;
    uint8_t lastByte;
    do {
      if (offset == 0)
        return emitError("invalid inverted VBR");
      lastByte = contents[offset - 1];
      result = (result << 7) | (lastByte & 0x7F);
      --offset;
    } while (lastByte & 0x80);
    return false;
  }

  /// Skip over bytes until reaching the specified alignment.
  bool alignTo(unsigned alignment) {
    if (!llvm::isPowerOf2_32(alignment))
      return emitError(Twine("tried to align to non-power-of-two: ") +
                       Twine(alignment));

    uint8_t padding;
    while (uintptr_t(contents.data()) & (uintptr_t(alignment) - 1)) {
      if (parseByte(padding))
        return true;
      if (padding != MEF::kAlignmentByte)
        return emitError("expected alignment byte, got '" +
                         Twine(int(padding)) + "' instead");
    }

    return false;
  }

  /// Take some number of bytes from the input file and put them into result.
  bool parseBytes(size_t length, ArrayRef<uint8_t> &result) {
    if (length > contents.size())
      return emitError("missing block of data");
    result = contents.take_front(length);
    contents = contents.drop_front(length);
    return false;
  }

  // Parse a single float value and put it into result.
  __attribute__((no_sanitize("undefined"))) bool parseFloat(float &result) {
    ArrayRef<uint8_t> scale;
    if (parseBytes(sizeof(float), scale))
      return emitError("couldn't parse float");
    result = *(const float *)scale.data();
    return false;
  }

  /// Ignore the first `length` bytes in this reader.
  bool dropBytes(size_t length) {
    ArrayRef<uint8_t> unused;
    return parseBytes(length, unused);
  }

  /// Parse a NUL terminated string into the specified StringRef, not including
  /// the NUL terminator.
  bool parseNULTerminatedString(StringRef &result) {
    auto *start = (const char *)contents.data();
    auto nul = (const char *)memchr(start, 0, contents.size());
    if (!nul)
      return emitError("malformed NUL terminated string");
    result = StringRef(start, nul - start);
    contents = contents.drop_front(result.size() + 1);
    return false;
  }

  /// Parse a NUL terminated string into the specified StringRef, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(size_t position, StringRef &result) const {
    if (position > contents.size())
      return emitError("invalid offset to string");
    auto buffer = contents.drop_front(position);
    auto *start = (const char *)buffer.data();
    auto nul = (const char *)memchr(start, 0, buffer.size());
    if (!nul)
      return emitError("malformed NUL terminated string");
    result = StringRef(start, nul - start);
    return false;
  }

  /// Parse a section header and the alignment metadata, returning the kind of
  /// section in `sectionID` and the contents of the section in `sectionData`.
  bool readSection(uint8_t &sectionID, ArrayRef<uint8_t> &sectionData) {
    size_t length;
    if (parseByte(sectionID) || parseVBR(length))
      return true;
    if (sectionID == (unsigned)MEF::SectionID::kInvalid)
      return emitError("invalid section ID");

    // The low bit of the size is a boolean indicating whether there is an
    // alignment byte + padding present.
    if ((length & 1) != 0) {
      // If there is an alignment byte, read the alignment byte, and skip over
      // padding bytes until we reach the desired alignment.
      uint8_t alignment = 0;
      if (parseByte(alignment) || alignTo(alignment))
        return true;
    }

    // Okay, the returned data is whatever is left now.  We shift by one to
    // eliminate the alignment bit.
    return parseBytes(length >> 1, sectionData);
  }

private:
  ArrayRef<uint8_t> contents;
  llvm::unique_function<void(Error) const> errorFunction;
};

} // namespace M

#endif // MLRT_GRAPHRT_MEF_MEFREADER_H
