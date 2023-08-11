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
  std::replace_if(
      out.begin(), out.end(), [](char c) { return c == '+'; }, '-');
  std::replace_if(
      out.begin(), out.end(), [](char c) { return c == '/'; }, '_');
  return out;
}

/// Replace the url-safe characters, then decode with LLVM's normal decoding
/// method.
ErrorOr<std::string> M::decodeURLSafeBase64(StringRef str) {
  std::string out = str.str();
  std::replace_if(
      out.begin(), out.end(), [](char c) { return c == '-'; }, '+');
  std::replace_if(
      out.begin(), out.end(), [](char c) { return c == '_'; }, '/');

  std::vector<char> output;
  auto err = llvm::decodeBase64(out, output);
  if (err)
    return Error(toString(std::move(err)));

  return std::string(output.begin(), output.end());
}
