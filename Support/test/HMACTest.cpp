//===- HMACTest.cpp -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HMAC.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

/// Compute the HMAC of some test vectors provided by the wikipedia article on
/// HMAC.
int main() {
  auto output =
      hmacSHA256("The quick brown fox jumps over the lazy dog", "key");
  auto hexStr = llvm::toHex(output);
  if (!StringRef(hexStr).equals_insensitive(
          "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8")) {
    llvm::outs() << "HMAC mismatch for short key: " << hexStr;
    return EXIT_FAILURE;
  }

  output =
      hmacSHA256("message", "The quick brown fox jumps over the lazy dogThe "
                            "quick brown fox jumps over the lazy dog");
  hexStr = llvm::toHex(output);
  if (!StringRef(hexStr).equals_insensitive(
          "5597b93a2843078cbb0c920ae41dfe20f1685e10c67e423c11ab91adfc319d12")) {
    llvm::outs() << "HMAC mismatch for long key: " << hexStr;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
