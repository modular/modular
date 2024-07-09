//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/NameMangling.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"

using namespace M;
using namespace KGEN;

/// Return whether the character is valid. Alnum, underscore, and period
/// characters are valid.
static constexpr bool isValid(char c) { return c == '_' || std::isalnum(c); }

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

namespace {
/// `std::pair<char, char>` is not constexpr apparently.
struct TwoChars {
  char d0, d1;
};
} // namespace

/// Encode each invalid character as a pair of valid characters.
static constexpr auto produceEncoding() {
  auto cipher = produceCipher();
  static_assert(cipher.size() * cipher.size() >= 256,
                "not enough valid characters");
  std::array<TwoChars, 256> encoding = {};
  for (int c = 0; c < 256; ++c)
    encoding[c] = {cipher[c % cipher.size()], cipher[c / cipher.size()]};
  return encoding;
}

/// Replace contiguous sections of invalid symbols with a single '_' while
/// tallying all the invalid symbols. They are encoded and placed at the end of
/// the string.
/// The resultant string. Each invalid character is encoded as 2 characters
/// additional characters and replaced with at most 1 underscore, meaning the
/// resulting string will be at most 3 times the size of the input string.
static SmallString<1024> replaceInvalidCharacter(StringRef name) {

  SmallVector<char, 256> invalid;
  invalid.reserve(name.size());
  SmallString<1024> result;
  result.reserve(name.size() * 3);
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
    invalid.push_back(c);
    carryingInvalid = true;
  }
  if (invalid.empty())
    return name;
  static constexpr auto encoding = produceEncoding();
  for (char c : invalid) {
    auto [d0, d1] = encoding[c];
    result.push_back(d0);
    result.push_back(d1);
  }
  return result;
}

StringAttr KGEN::sanitizeSymbolToAlnum(StringAttr name, size_t charToKeep) {
  VerboseCompilerTimeTraceScope traceScope("sanitizeSymbolToAlnum",
                                           [name] { return name.str(); });
  if (name.size() > charToKeep) {
    auto rawNameBytes =
        ArrayRef<uint8_t>((const uint8_t *)name.data(), name.size());
    auto hash = llvm::BLAKE3::hash<16>(rawNameBytes);
    return StringAttr::get(
        name.getContext(),
        replaceInvalidCharacter(name.strref().take_front(charToKeep)) + "_" +
            llvm::toHex(hash, true));
  }

  return StringAttr::get(name.getContext(), replaceInvalidCharacter(name));
}
