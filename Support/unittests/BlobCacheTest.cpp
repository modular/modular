//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BlobCache.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

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

class BlobCacheTest : public testing::Test {
protected:
  BlobCache<StringKeyInfo> cache{getDefaultBackendChain("")};
};

} // namespace

TEST_F(BlobCacheTest, NotContainItemThatHasNotBeenInserted) {
  EXPECT_FALSE(cache.contains("does not exist"))
      << "expected not to have item named 'does not exist'\n";
}

TEST_F(BlobCacheTest, FindShouldReturnErrorForNonexistantItem) {
  auto dneOr = cache.find("does not exist");
  EXPECT_FALSE(dneOr.hasValue() && !dneOr.isError())
      << "expected not to have item named 'does not exist'\n";
}

TEST_F(BlobCacheTest, ContainItemWhenInserted) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosBuf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(32);

  ErrorOr<std::string> err = cache.insert("zeros", *zerosBuf);
  EXPECT_FALSE(err.isError()) << err.getError() << '\n';
  EXPECT_FALSE(err->empty()) << "expected to receive the hash key\v";
  EXPECT_TRUE(cache.contains("zeros"))
      << "expected to have item named 'zeros'\n";
}

TEST_F(BlobCacheTest, FindItemThatExists) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosBuf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(32);

  auto err = cache.insert("zeros", *zerosBuf);
  ASSERT_FALSE(failed(err)) << err.getError() << '\n';

  auto zerosOr = cache.find("zeros");
  EXPECT_TRUE(zerosOr.hasValue()) << zerosOr.getError();

  std::unique_ptr<llvm::MemoryBuffer> outZeros = zerosOr.takeValue();
  ASSERT_TRUE(outZeros->getBufferSize() == zerosBuf->getBufferSize())
      << "output buffer size did not match input buffer size\n";
  EXPECT_TRUE(outZeros->getBuffer() ==
              StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize()))
      << "buffer returned did not match the buffer inputted\n";
}

/* TODO: Disabled as part of #4394.
TEST_F(BlobCacheTest, FileSystemFindItemThatExists) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosBuf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(32);

  auto err = cache.insert("zeros", *zerosBuf);
  ASSERT_FALSE(failed(err)) << err.getError() << '\n';

  // Reset the cache so that we are forced to look it up from the file system.
  cache = BlobCache<StringKeyInfo>(getDefaultBackendChain(""));

  // Check that the cache holds the new item, and it's the same data as before.
  auto zerosOr = cache.find("zeros");
  EXPECT_TRUE(zerosOr.hasValue()) << zerosOr.getError();

  std::unique_ptr<llvm::MemoryBuffer> outZeros = zerosOr.takeValue();
  ASSERT_TRUE(outZeros->getBufferSize() == zerosBuf->getBufferSize())
      << "output buffer size did not match input buffer size\n";
  EXPECT_TRUE(outZeros->getBuffer() ==
              StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize()))
      << "buffer returned did not match the buffer inputted\n";
}
*/
