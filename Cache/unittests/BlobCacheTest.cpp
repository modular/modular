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
            getLocalDefaultBackendChain(runtime, STRINGIFY(CACHE_TEST_DIR))
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
    EXPECT_FALSE(dneOr.isError())
        << "expected to not have an error for unknown item\n";
    EXPECT_FALSE(dneOr->has_value())
        << "expected not to have item named 'does not exist'\n";
  });
}

TEST_F(BlobCacheTest, ContainItemWhenInserted) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> insertOr =
      cache->insert("zeros", std::move(zerosBuf));
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr.isError())
        << insertOr.getDiagnostic().getMessage() << '\n';
    EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

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

  AsyncValueRef<std::string> insertOr = cache->insert("zeros", zerosBuf.copy());
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr.isError())
        << insertOr.getDiagnostic().getMessage() << '\n';
    EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

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

TEST_F(BlobCacheTest, FindItemThatExistsWithPreallocatedBuf) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> insertOr = cache->insert("zeros", zerosBuf.copy());
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr.isError())
        << insertOr.getDiagnostic().getMessage() << '\n';
    EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

    auto contains = cache->contains("zeros");
    contains.andThenSync([contains = contains.copy()] {
      EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
    });
  });

  // Get a buffer to read into.
  auto readBuf = WriteableBuffer::get(32);

  auto zerosOr = cache->find("zeros", readBuf.copy());
  zerosOr.andThenSync([zerosOr = zerosOr.copy(), zerosBuf = zerosBuf.copy(),
                       readBuf = std::move(readBuf)]() mutable {
    EXPECT_TRUE(zerosOr->has_value()) << zerosOr.getDiagnostic().getMessage();
    ASSERT_TRUE(readBuf->getBufferSize() == 32)
        << "output buffer size did not match expected buffer size\n";
    EXPECT_TRUE(readBuf->getBuffer()[0] == '\0')
        << "buffer returned did not match the buffer inputted\n";
  });
}

TEST_F(BlobCacheTest, FindItemThatExistsThenClear) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> insertOr = cache->insert("zeros", zerosBuf.copy());
  insertOr.andThenSync([cache = cache.copy(), insertOr = insertOr.copy()] {
    EXPECT_FALSE(insertOr.isError())
        << insertOr.getDiagnostic().getMessage() << '\n';
    EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

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
    EXPECT_FALSE(clearOr.isError())
        << clearOr.getDiagnostic().getMessage() << "\n";

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

  AsyncValueRef<std::string> err = cache->insert("zeros", zerosBuf.copy());
  await(err);
  ASSERT_FALSE(err.isError()) << err.getDiagnostic().getMessage() << '\n';

  // Reset the cache so that we are forced to look it up from the file system.
  auto fsCache = LLCL::RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(runtime, STRINGIFY(CACHE_TEST_DIR))
          .takeValue());

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

TEST_F(BlobCacheTest, FileSystemFindItemThatExistsWithPreallocatedBuffer) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> err = cache->insert("zeros", zerosBuf.copy());
  await(err);
  ASSERT_FALSE(err.isError()) << err.getDiagnostic().getMessage() << '\n';

  // Reset the cache so that we are forced to look it up from the file system.
  auto fsCache = LLCL::RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(runtime, STRINGIFY(CACHE_TEST_DIR))
          .takeValue());

  // Get a buffer to read into.
  auto readBuf = WriteableBuffer::get(32);
  const char *readBufStart = readBuf->getBufferStart();

  // Check that the cache holds the new item, and it's the same data as before.
  auto zerosOr = fsCache->find("zeros", readBuf.copy());
  await(zerosOr);
  EXPECT_TRUE(zerosOr->has_value()) << zerosOr.getDiagnostic().getMessage();

  ASSERT_TRUE(readBuf->getBufferSize() == 32)
      << "output buffer size did not match input buffer size\n";
  EXPECT_TRUE(readBuf->getBuffer()[0] == '\0')
      << "buffer returned did not match the buffer input\n";

  // Do the find again, and ensure that the read buffer's pointer hasn't
  // changed (it should have hit the in-memory cache, which should be literally
  // holding a read-only reference to the buffer).
  auto zerosOrAgain = fsCache->find("zeros", readBuf.copy());
  await(zerosOrAgain);
  EXPECT_TRUE(zerosOrAgain->has_value())
      << zerosOr.getDiagnostic().getMessage();

  ASSERT_TRUE(readBuf->getBufferSize() == 32)
      << "output buffer size did not match input buffer size\n";
  ASSERT_TRUE(readBuf->getBufferStart() == readBufStart)
      << "read buffer pointer changed\n";
  EXPECT_TRUE(readBuf->getBuffer()[0] == '\0')
      << "buffer returned did not match the buffer input\n";
}

TEST_F(BlobCacheTest, FileSystemTestOldVersionDeletion) {
  // Mock the existence of an old version of the cache.
  // Specifically create the directory to have a trailing path separator to
  // test canonicalization of paths when figuring out deletion criteria.
  auto cacheDir = std::filesystem::path(STRINGIFY(CACHE_TEST_DIR)) / "";
  auto tempDirectory = cacheDir / "ModularOldVersionString";

  std::error_code ec;
  std::filesystem::create_directory(tempDirectory, ec);
  ASSERT_FALSE(ec) << "failed to create directory: " << ec.message() << "\n";

  // Upon creating a new cache, all of the old versions on the filesystem
  // should be deleted.
  LLCL::Runtime runtime(createLeakCheckAllocator(createMallocAllocator()),
                        createThreadPoolWorkQueue());
  auto fsCache = LLCL::RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(runtime, cacheDir).takeValue());
  ASSERT_TRUE(!std::filesystem::exists(tempDirectory))
      << "expected the temp directory to be deleted by cacheDir creation\n";
}
