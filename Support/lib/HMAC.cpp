//===- HMAC.cpp -----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HMAC.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

using namespace M;

/// These are static constants from
/// https://cseweb.ucsd.edu//~mihir/papers/hmac-cb.pdf
static constexpr uint8_t innerPaddingByte = 0x36;
static constexpr uint8_t outerPaddingByte = 0x5c;
static constexpr size_t sha256BlockSize = 64;
using SHA256State = std::array<uint8_t, sha256BlockSize>;

/// Do a classic HMAC using SHA256 as the hash function.
SHA256Hash M::hmacSHA256(StringRef data, StringRef key) {
  // First normalize the key.
  SHA256State keyBytes = {
      0,
  };
  // If the key is larger than sha256Bytes then hash it and put it at the
  // beginning.
  if (key.size() > sha256BlockSize) {
    auto hashed = llvm::SHA256::hash(
        llvm::makeArrayRef(key.bytes_begin(), key.bytes_end()));
    std::copy(hashed.begin(), hashed.end(), keyBytes.begin());
  } else {
    std::copy(key.begin(), key.end(), keyBytes.begin());
  }

  // Compute the inner and outer padded keys.
  SHA256State innerKeyBytes, outerKeyBytes;
  for (size_t i = 0; i < sha256BlockSize; ++i) {
    innerKeyBytes[i] = keyBytes[i] ^ innerPaddingByte;
    outerKeyBytes[i] = keyBytes[i] ^ outerPaddingByte;
  }

  llvm::SHA256 sha;
  sha.init();
  // First we hash the concatenation of the inner key and the message.
  sha.update(innerKeyBytes);
  sha.update(data);
  SHA256Hash innerKeyAndMessage = sha.result();
  // Re-init the hash function's internal state.
  sha.init();
  // Next we hash the outer key concatenated with the previous step.
  sha.update(outerKeyBytes);
  sha.update(innerKeyAndMessage);
  // This is the HMAC.
  return sha.result();
}
