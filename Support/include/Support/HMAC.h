//===- Support/HMAC.h -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HMAC_H
#define SUPPORT_HMAC_H

#include "Support/LLVMForwardDecls.h"
#include <array>
#include <cstdint>

/// HMAC is a keyed hash construction that allows integrity and identity
/// checking. There are various signature-related use cases for an HMAC, but the
/// purpose of the functions in this file is not that (as of Aug 29, 2022). The
/// purpose we're interested in is integrity checking (has the file been
/// tampered-with/is the file corrupted?) and identification (is the file one of
/// ours?). HMAC solves both problems - being a keyed hash you can check "was
/// this generated from something with the key" (identification) and "has this
/// been corrupted" (integrity) at once.

namespace M {
/// Useful definitions for working with SHA256.
static constexpr size_t sha256Bytes = 256 / 8;
using SHA256Hash = std::array<uint8_t, sha256Bytes>;

/// This implements a simple low-dependency HMAC-SHA-256. Note that this has not
/// been cryptanalzyed and so should not be used for security applications!
SHA256Hash hmacSHA256(StringRef data, StringRef key);

} // namespace M

#endif // SUPPORT_HMAC_H
