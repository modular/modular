//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HMAC.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/SHA256.h"

using namespace M;

/// These are static constants from
/// https://cseweb.ucsd.edu//~mihir/papers/hmac-cb.pdf
static constexpr uint8_t innerPaddingByte = 0x36;
static constexpr uint8_t outerPaddingByte = 0x5c;
static constexpr size_t hashBlockSize = 64;
using HashState = std::array<uint8_t, hashBlockSize>;

template <typename ReturnT, typename HasherT>
static ReturnT doHash(StringRef data, StringRef key, HasherT &&hasher) {
  // First normalize the key.
  HashState keyBytes = {
      0,
  };
  // If the key is larger than sha256Bytes then hash it and put it at the
  // beginning.
  if (key.size() > hashBlockSize) {
    auto hashed =
        llvm::SHA256::hash(ArrayRef(key.bytes_begin(), key.bytes_end()));
    std::copy(hashed.begin(), hashed.end(), keyBytes.begin());
  } else {
    std::copy(key.begin(), key.end(), keyBytes.begin());
  }

  // Compute the inner and outer padded keys.
  HashState innerKeyBytes, outerKeyBytes;
  for (size_t i = 0; i < hashBlockSize; ++i) {
    innerKeyBytes[i] = keyBytes[i] ^ innerPaddingByte;
    outerKeyBytes[i] = keyBytes[i] ^ outerPaddingByte;
  }

  hasher.init();
  // First we hash the concatenation of the inner key and the message.
  hasher.update(innerKeyBytes);
  hasher.update(data);
  ReturnT innerKeyAndMessage = hasher.result();
  // Re-init the hash function's internal state.
  hasher.init();
  // Next we hash the outer key concatenated with the previous step.
  hasher.update(outerKeyBytes);
  hasher.update(innerKeyAndMessage);
  // This is the HMAC.
  return hasher.result();
}

/// Do a classic HMAC using SHA256 as the hash function.
SHA256Hash M::hmacSHA256(StringRef data, StringRef key) {
  llvm::SHA256 sha;
  return doHash<SHA256Hash>(data, key, sha);
}

/// Do a classic HMAC using BLAKE3 as the hash function.
BLAKE3Hash M::hmacBLAKE3(StringRef data, StringRef key) {
  llvm::BLAKE3 blake;
  return doHash<BLAKE3Hash>(data, key, blake);
}
