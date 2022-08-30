//===- BlobCacheTest.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BlobCache.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

namespace {
/// Basic string key info.
struct StringKeyInfo {
  using KeyTy = StringRef;

  static std::string hashKey(KeyTy key) {
    ArrayRef<uint8_t> bytes((const uint8_t *)key.data(), key.size());
    return llvm::toHex(llvm::SHA256::hash(bytes), true);
  }
};
} // namespace

int main() {
  BlobCache<StringKeyInfo> cache(getDefaultBackendChain(""));

  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosBuf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(32);

  if (auto err = cache.insert("zeros", *zerosBuf)) {
    llvm::outs() << err.getError() << "\n";
    return EXIT_FAILURE;
  }

  if (!cache.contains("zeros")) {
    llvm::outs() << "expected to have item named 'zeros'\n";
    return EXIT_FAILURE;
  }

  if (cache.contains("does not exist")) {
    llvm::outs() << "expected not to have item named 'does not exist'\n";
    return EXIT_FAILURE;
  }

  auto zerosOr = cache.find("zeros");
  if (failed(zerosOr)) {
    llvm::outs() << zerosOr.getError() << "\n";
    return EXIT_FAILURE;
  }

  auto dneOr = cache.find("does not exist");
  if (succeeded(dneOr)) {
    llvm::outs() << "expected not to have item named 'does not exist'\n";
    return EXIT_FAILURE;
  }

  if ((*zerosOr)->getBufferSize() != zerosBuf->getBufferSize()) {
    llvm::outs() << "output buffer size did not match input buffer size\n";
    return EXIT_FAILURE;
  }

  if ((*zerosOr)->getBuffer() !=
      StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize())) {
    llvm::outs() << "buffer returned did not match the buffer inputted\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
