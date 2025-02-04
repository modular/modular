//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <string>
#include <vector>

#include "Support/Entitlements/EntitlementToken.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/JSON.h"

using namespace M;

ErrorOr<std::unique_ptr<EntitlementToken>>
M::unpackToken(llvm::StringRef b64token) {
  std::vector<char> rawstr;
  rawstr.resize(b64token.size());
  auto err = llvm::decodeBase64(b64token, rawstr);
  if (err) {
    return Error("unable to b64 decode");
  }
  std::string jsonString(rawstr.begin(), rawstr.end());

  // uncompress
  if (!llvm::compression::zlib::isAvailable()) {
    return Error("No zlib");
  }

  // convert vector<char> to vector<uint8> to accomodate zlib.decompress
  std::vector<uint8_t> compressed;
  compressed.resize(jsonString.size());
  std::copy_n(jsonString.begin(), jsonString.size(), compressed.begin());

  // Uncompressed payloads should be a few kB, Allow payloads up to 1 MB.
  size_t size = 1000000;
  llvm::SmallVector<uint8_t> uncompressed;
  llvm::Error decompErr = llvm::compression::zlib::decompress(
      llvm::ArrayRef<uint8_t>(compressed), uncompressed, size);
  if (decompErr) {
    return Error("failed to decompress: " +
                 llvm::toString(std::move(decompErr)));
  }

  std::string jsonString2(uncompressed.begin(), uncompressed.end());

  auto expVal = llvm::json::parse(jsonString2);
  if (!expVal) {
    llvm::consumeError(expVal.takeError());
    return Error("unable to parse json: ");
  }
  auto val = expVal.get();
  llvm::json::Object *obj = val.getAsObject();

  std::unique_ptr<EntitlementToken> t = std::make_unique<EntitlementToken>();

  auto keyOr = obj->getString("key");
  if (!keyOr) {
    return Error("no key");
  }
  t->key = *keyOr;

  const llvm::json::Array *chain = obj->getArray("cert_chain");
  if (chain == nullptr) {
    return Error("no cert_chain");
  }
  t->certChain.reserve(chain->size());
  for (const auto &pemVal : *chain) {
    auto pemOr = pemVal.getAsString();
    if (!pemOr) {
      return Error("cert chain value not string");
    }
    t->certChain.emplace_back(*pemOr);
  }

  return t;
}

std::string M::packToken(const EntitlementToken &token) {
  std::string jsonstr;
  llvm::raw_string_ostream os(jsonstr);
  llvm::json::OStream j(os);

  j.object([&] {
    j.attributeArray("cert_chain", [&] {
      for (auto &pem : token.certChain) {
        j.value(pem);
      }
    });
    j.attribute("key", token.key);
  });

  j.flush();

  // copy jsonstr to a vector<uint_8> to acommodate zlib::compress
  std::vector<uint8_t> uncompressed;
  uncompressed.resize(jsonstr.size());
  std::copy_n(jsonstr.begin(), jsonstr.size(), uncompressed.begin());

  // compress
  llvm::SmallVector<uint8_t> compressed;
  llvm::compression::zlib::compress(uncompressed, compressed);

  return llvm::encodeBase64(compressed);
}
