//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BASE64_H
#define SUPPORT_BASE64_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include <string>

namespace M {
// TODO: We should contribute better Base64 encoding/decoding to upstream, this
//       should not be necessary.

/// Base-64 encode a string that conforms to RFC4648 Section 5 (URL and filename
/// safe). This implementation does not include the `=` padding at the end of
/// the encoded bytes.
std::string encodeURLSafeBase64(StringRef str);

/// Base-64 decode a string that conforms to RFC4648 Section 5 (URL and filename
/// safe). This implementation does not include the `=` padding at the end of
/// the encoded bytes.
ErrorOr<std::string> decodeURLSafeBase64(StringRef str);
} // namespace M

#endif // SUPPORT_BASE64_H
