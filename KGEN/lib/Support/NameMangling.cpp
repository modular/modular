//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/NameMangling.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace M;
using namespace KGEN;

/// Produce an array of all the valid characters. This array will be used to
/// encode the unsupported characters.
static constexpr auto produceCipher() {
  // alnum + underscore is twice the alphabet, each digit, and the underscore.
  constexpr size_t size = 26 * 2 + 10 + 1;
  std::array<char, size> cipher = {};
  unsigned i = 0;
  auto fill = [&](char lb, char ub) {
    for (char c = lb; c <= ub; ++c)
      cipher[i++] = c;
  };
  fill('A', 'Z');
  fill('a', 'z');
  fill('0', '9');
  cipher[i] = '_';
  return cipher;
}

StringAttr KGEN::sanitizeSymbolToAlnum(StringAttr name) {
  // Replace contiguous sections of invalid symbols with a single '_' while
  // tallying all the invalid symbols.
  //
  // This algorithm will iterate over `name` while appending allowed characters
  // to `result` and using a flag `carryingInvalid` to replace ranges of invalid
  // characters with a single underscore. At the same type, the bytes of all
  // invalid characters are appended together to from a big integer. The big
  // integer is then encoded using a cipher of the allowed characters.
  auto isValid = [](char c) { return c == '_' || c == '.' || std::isalnum(c); };

  // APInt accepts a constructor of an array of 64-bit integer "segments", so
  // build the big int by filling in this vector.
  SmallVector<uint64_t> invalid;

  // The current big int segment being built.
  uint64_t curSegment = 0;
  // The byte offset into the current segment to be filled with a character
  // next.
  uint8_t curOffset = 0;

  // The resultant string.
  std::string result;
  result.reserve(name.size() * 2);
  bool carryingInvalid = false;
  for (char c : name) {
    if (isValid(c)) {
      // If the last character was invalid, push an underscore.
      if (carryingInvalid) {
        carryingInvalid = false;
        result.push_back('_');
      }
      // Push the valid character.
      result.push_back(c);
      continue;
    }
    carryingInvalid = true;
    // Overflow. Push the complete segment onto the vector and start a new one.
    if (curOffset >= 8) {
      invalid.push_back(curSegment);
      curOffset = 0;
      curSegment = 0;
    }
    // Write the byte into the current big int segment at the current offset.
    curSegment |= ((uint64_t)c) << (curOffset++ * 8);
  }
  // If any invalid characters were found, a byte would have been written and
  // `curOffset` will not be zero. If nothing changed, just return the result.
  if (!curOffset) {
    assert(result == name && "no invalid symbols");
    return name;
  }
  // `carryingInvalid` doesn't matter since we will always add a '_' here.
  result.push_back('_');
  // Add the last segment in.
  invalid.push_back(curSegment);

  // Use the cipher to encode the invalid characters at the end of the symbol.
  // Compute the number of bits accounting for the fact that the last segment
  // may have been partial.
  APInt bigVal((invalid.size() - 1) * 64 + curOffset * 8, invalid);
  constexpr auto cipher = produceCipher();
  // Run the loop at least once, because zero should be encoded.
  do {
    // Compute the next valid character to push onto the string.
    APInt quotient;
    uint64_t remainder;
    APInt::udivrem(bigVal, cipher.size(), quotient, remainder);
    result.push_back(cipher[remainder]);
    bigVal = std::move(quotient);
  } while (!bigVal.isZero());
  return StringAttr::get(name.getContext(), result);
}
