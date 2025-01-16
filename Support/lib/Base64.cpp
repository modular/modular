//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Base64.h"
#include "llvm/Support/Base64.h"

using namespace M;

/// Use LLVM's normal encoding method, then replace with the URL-safe
/// characters.
std::string M::encodeURLSafeBase64(StringRef str) {
  std::string out = llvm::encodeBase64(str);
  std::replace_if(out.begin(), out.end(), [](char c) { return c == '+'; }, '-');
  std::replace_if(out.begin(), out.end(), [](char c) { return c == '/'; }, '_');
  // Only remove up to 3 padding '='.
  for (size_t i = 0; i < 3; ++i) {
    if (out.back() != '=')
      break;
    out.pop_back();
  }
  return out;
}

/// Replace the url-safe characters, then decode with LLVM's normal decoding
/// method.
ErrorOr<std::string> M::decodeURLSafeBase64(StringRef str) {
  std::string out = str.str();
  std::replace_if(out.begin(), out.end(), [](char c) { return c == '-'; }, '+');
  std::replace_if(out.begin(), out.end(), [](char c) { return c == '_'; }, '/');

  // Add back in any padding we removed, but only if necessary.
  size_t remainder = str.size() % 4;
  if (remainder != 0)
    out += std::string(4 - remainder, '=');

  std::vector<char> output;
  auto err = llvm::decodeBase64(out, output);
  if (err)
    return toModularError(std::move(err));

  return std::string(output.begin(), output.end());
}
