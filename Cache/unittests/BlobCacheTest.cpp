//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/RCRef.h"
#include "Support/Preprocessor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

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
  LLCL::Runtime runtime;
  LLCL::RCRef<BlobCache<StringKeyInfo>> cache;
  BlobCacheTest()
      : runtime(createLeakCheckAllocator(createMallocAllocator()),
                createThreadPoolWorkQueue()),
        cache(LLCL::RCRef<BlobCache<StringKeyInfo>>::create(
            getDefaultBackendChain(runtime, STRINGIFY(CACHE_TEST_DIR))
                .takeValue())) {}
};

} // namespace

TEST_F(BlobCacheTest, NotContainItemThatHasNotBeenInserted) {
  auto contains = cache->contains("does not exist");
  contains.andThenSync([contains = contains.copy()] {
    EXPECT_FALSE(*contains)
        << "expected not to have item named 'does not exist'\n";
  });
}

TEST_F(BlobCacheTest, FindShouldNotReturnErrorForNonexistantItem) {
  auto dneOr = cache->find("does not exist");
  dneOr.andThenSync([dneOr = dneOr.copy()] {
    EXPECT_FALSE(dneOr->has_value() && !dneOr.isError())
        << "expected not to have item named 'does not exist'\n";
  });
}

TEST_F(BlobCacheTest, ContainItemWhenInserted) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto insertOr = cache->insert("zeros", std::move(zerosBuf));
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr->isError()) << insertOr->getError() << '\n';
    EXPECT_FALSE(insertOr->takeValue().empty())
        << "expected to receive the hash key\v";

    auto contains = cache->contains("zeros");
    contains.andThenSync([contains = contains.copy()] {
      EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
    });
  });
}

TEST_F(BlobCacheTest, FindItemThatExists) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto insertOr = cache->insert("zeros", zerosBuf.copy());
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr->isError()) << insertOr->getError() << '\n';
    EXPECT_FALSE(insertOr->takeValue().empty())
        << "expected to receive the hash key\v";

    auto contains = cache->contains("zeros");
    contains.andThenSync([contains = contains.copy()] {
      EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
    });
  });

  auto zerosOr = cache->find("zeros");
  zerosOr.andThenSync([zerosOr = zerosOr.copy(), zerosBuf = zerosBuf.copy()] {
    EXPECT_TRUE(zerosOr->has_value()) << zerosOr.getDiagnostic().getMessage();
    BufferRef outZeros = std::move(**zerosOr);
    ASSERT_TRUE(outZeros->getBufferSize() == zerosBuf->getBufferSize())
        << "output buffer size did not match input buffer size\n";
    EXPECT_TRUE(outZeros->getBuffer() == StringRef(zerosBuf->getBufferStart(),
                                                   zerosBuf->getBufferSize()))
        << "buffer returned did not match the buffer inputted\n";
  });
}

TEST_F(BlobCacheTest, FindItemThatExistsThenClear) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto insertOr = cache->insert("zeros", zerosBuf.copy());
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr->isError()) << insertOr->getError() << '\n';
    EXPECT_FALSE(insertOr->takeValue().empty())
        << "expected to receive the hash key\v";

    auto contains = cache->contains("zeros");
    contains.andThenSync([contains = contains.copy()] {
      EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
    });
  });
  await(insertOr);

  auto zerosOr = cache->find("zeros");
  zerosOr.andThenSync([zerosOr = zerosOr.copy(), zerosBuf = zerosBuf.copy()] {
    EXPECT_TRUE(zerosOr->has_value()) << zerosOr.getDiagnostic().getMessage();
    BufferRef outZeros = std::move(**zerosOr);
    ASSERT_TRUE(outZeros->getBufferSize() == zerosBuf->getBufferSize())
        << "output buffer size did not match input buffer size\n";
    EXPECT_TRUE(outZeros->getBuffer() == StringRef(zerosBuf->getBufferStart(),
                                                   zerosBuf->getBufferSize()))
        << "buffer returned did not match the buffer inputted\n";
  });
  // We have to sequence the clear *after* all the other work has been done. Use
  // await to make this more readable.
  await(zerosOr);

  auto clearOr = cache->clear();
  clearOr.andThenSync([clearOr = clearOr.copy(), cache = cache.copy()] {
    EXPECT_FALSE(failed(*clearOr)) << clearOr->getError() << "\n";

    auto contains = cache->contains("zeros");
    contains.andThenSync([contains = contains.copy()] {
      EXPECT_FALSE(*contains)
          << "expected not to have item named 'zeros' after the clear\n";
    });
  });
}

TEST_F(BlobCacheTest, FileSystemFindItemThatExists) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto err = cache->insert("zeros", zerosBuf.copy());
  await(err);
  ASSERT_FALSE(failed(*err)) << err->getError() << '\n';

  // Reset the cache so that we are forced to look it up from the file system.
  auto fsCache = LLCL::RCRef<BlobCache<StringKeyInfo>>::create(
      getDefaultBackendChain(runtime, STRINGIFY(CACHE_TEST_DIR)).takeValue());

  // Check that the cache holds the new item, and it's the same data as before.
  auto zerosOr = fsCache->find("zeros");
  await(zerosOr);
  EXPECT_TRUE(zerosOr->has_value()) << zerosOr.getDiagnostic().getMessage();

  BufferRef outZeros = std::move(**zerosOr);
  ASSERT_TRUE(outZeros->getBufferSize() == zerosBuf->getBufferSize())
      << "output buffer size did not match input buffer size\n";
  EXPECT_TRUE(outZeros->getBuffer() ==
              StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize()))
      << "buffer returned did not match the buffer inputted\n";
}
