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
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/FileSystemExtras.h"
#include "Support/Preprocessor.h"
#include "Support/RCRef.h"
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

static TempDir createTempDir() {
  auto tempDirOr = TempDir::create("cache-test.%%%%%%");
  assert(!tempDirOr.isError());
  return tempDirOr.takeValue();
}

class BlobCacheTest : public testing::Test {
protected:
  TempDir tempDir;
  std::unique_ptr<LLCL::Runtime> runtime;
  RCRef<BlobCache<StringKeyInfo>> cache;
  BlobCacheTest()
      : tempDir(createTempDir()),
        runtime(createUniqueRuntime(
            LLCL::RuntimeOptions().withLeakCheckedAllocator())),
        cache(RCRef<BlobCache<StringKeyInfo>>::create(
            getLocalDefaultBackendChain(*runtime, tempDir.getPath())
                .takeValue())) {}
};

} // namespace

TEST_F(BlobCacheTest, NotContainItemThatHasNotBeenInserted) {
  auto done = AsyncValueRef<Chain>::allocate(*runtime);
  auto contains = cache->contains("does not exist");
  std::move(contains).andThenSync(
      [done = done.copy()](AsyncValueRef<bool> contains) mutable {
        ASSERT_FALSE(contains.isError())
            << contains.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(*contains)
            << "expected not to have item named 'does not exist'\n";
        std::move(done).emplace();
      });
  await(done);
}

TEST_F(BlobCacheTest, FindShouldNotReturnErrorForNonexistantItem) {
  auto done = AsyncValueRef<Chain>::allocate(*runtime);
  auto dneOr = cache->find("does not exist");
  std::move(dneOr).andThenSync(
      [done =
           done.copy()](AsyncValueRef<std::optional<BufferRef>> dneOr) mutable {
        ASSERT_FALSE(dneOr.isError())
            << dneOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(dneOr->has_value())
            << "expected not to have item named 'does not exist'\n";
        std::move(done).emplace();
      });
  await(done);
}

TEST_F(BlobCacheTest, ContainItemWhenInserted) {
  auto done = AsyncValueRef<Chain>::allocate(*runtime);

  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> insertOr =
      cache->insert("zeros", std::move(zerosBuf));
  std::move(insertOr).andThenSync(
      [cache = cache.copy(),
       done = done.copy()](AsyncValueRef<std::string> insertOr) mutable {
        ASSERT_FALSE(insertOr.isError())
            << insertOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

        auto contains = cache->contains("zeros");
        std::move(contains).andThenSync(
            [done = std::move(done)](AsyncValueRef<bool> contains) mutable {
              ASSERT_FALSE(contains.isError())
                  << contains.getDiagnostic().getMessage() << '\n';
              EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
              std::move(done).emplace();
            });
      });
  await(done);
}

TEST_F(BlobCacheTest, FindItemThatExists) {

  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto inserted = AsyncValueRef<Chain>::allocate(*runtime);
  AsyncValueRef<std::string> insertOr = cache->insert("zeros", zerosBuf.copy());
  std::move(insertOr).andThenSync(
      [cache = cache.copy(), inserted = inserted.copy()](
          AsyncValueRef<std::string> insertOr) mutable {
        ASSERT_FALSE(insertOr.isError())
            << insertOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

        auto contains = cache->contains("zeros");
        std::move(contains).andThenSync(
            [inserted =
                 std::move(inserted)](AsyncValueRef<bool> contains) mutable {
              ASSERT_FALSE(contains.isError())
                  << contains.getDiagnostic().getMessage() << '\n';
              EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
              std::move(inserted).emplace();
            });
      });
  await(inserted);

  auto found = AsyncValueRef<Chain>::allocate(*runtime);
  auto zerosOr = cache->find("zeros");
  std::move(zerosOr).andThenSync(
      [zerosBuf = zerosBuf.copy(), found = found.copy()](
          AsyncValueRef<std::optional<BufferRef>> zerosOr) mutable {
        ASSERT_FALSE(zerosOr.isError())
            << zerosOr.getDiagnostic().getMessage() << '\n';
        ASSERT_TRUE(zerosOr->has_value());
        BufferRef outZeros = std::move(**zerosOr);
        ASSERT_EQ(outZeros->getBufferSize(), zerosBuf->getBufferSize())
            << "output buffer size did not match input buffer size\n";
        EXPECT_TRUE(
            outZeros->getBuffer() ==
            StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize()))
            << "buffer returned did not match the buffer inputted\n";
        std::move(found).emplace();
      });
  await(found);
}

TEST_F(BlobCacheTest, FindItemThatExistsWithPreallocatedBuf) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto inserted = AsyncValueRef<Chain>::allocate(*runtime);
  AsyncValueRef<std::string> insertOr = cache->insert("zeros", zerosBuf.copy());
  std::move(insertOr).andThenSync(
      [cache = cache.copy(), inserted = inserted.copy()](
          AsyncValueRef<std::string> insertOr) mutable {
        ASSERT_FALSE(insertOr.isError())
            << insertOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

        auto contains = cache->contains("zeros");
        std::move(contains).andThenSync(
            [inserted =
                 std::move(inserted)](AsyncValueRef<bool> contains) mutable {
              ASSERT_FALSE(contains.isError())
                  << contains.getDiagnostic().getMessage() << '\n';
              EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
              std::move(inserted).emplace();
            });
      });
  await(inserted);

  // Get a buffer to read into.
  auto readBuf = WriteableBuffer::get(/*size=*/0, /*alignment=*/std::nullopt,
                                      /*capacity=*/32);

  auto found = AsyncValueRef<Chain>::allocate(*runtime);
  auto zerosOr = cache->find("zeros", readBuf.copy());
  std::move(zerosOr).andThenSync(
      [zerosBuf = zerosBuf.copy(), readBuf = std::move(readBuf),
       found = found.copy()](
          AsyncValueRef<std::optional<BufferRef>> zerosOr) mutable {
        ASSERT_FALSE(zerosOr.isError())
            << zerosOr.getDiagnostic().getMessage() << '\n';
        ASSERT_TRUE(zerosOr->has_value());
        ASSERT_TRUE(readBuf->getBufferSize() == 1)
            << "output buffer size did not match expected buffer size\n";
        EXPECT_TRUE(readBuf->getBuffer()[0] == '\0')
            << "buffer returned did not match the buffer inputted\n";
        std::move(found).emplace();
      });
  await(found);
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
  auto fsCache = RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(*runtime, tempDir.getPath()).takeValue());

  // Check that the cache holds the new item, and it's the same data as before.
  auto zerosOr = fsCache->find("zeros");
  await(zerosOr);
  ASSERT_FALSE(zerosOr.isError())
      << zerosOr.getDiagnostic().getMessage() << '\n';
  ASSERT_TRUE(zerosOr->has_value());

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
  auto fsCache = RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(*runtime, tempDir.getPath()).takeValue());

  // Get a buffer to read into.
  auto readBuf = WriteableBuffer::get(/*size=*/0, /*alignment=*/std::nullopt,
                                      /*capacity=*/32);
  const char *readBufStart = readBuf->getBufferStart();

  // Check that the cache holds the new item, and it's the same data as before.
  auto zerosOr = fsCache->find("zeros", readBuf.copy());
  await(zerosOr);
  ASSERT_FALSE(zerosOr.isError())
      << zerosOr.getDiagnostic().getMessage() << '\n';
  EXPECT_TRUE(zerosOr->has_value());

  ASSERT_TRUE(readBuf->getBufferSize() == 1)
      << "output buffer size did not match input buffer size\n";
  EXPECT_TRUE(readBuf->getBuffer()[0] == '\0')
      << "buffer returned did not match the buffer input\n";

  // Do the find again, and ensure that the read buffer's pointer hasn't
  // changed (it should have hit the in-memory cache, which should be literally
  // holding a read-only reference to the buffer).
  auto zerosOrAgain = fsCache->find("zeros", readBuf.copy());
  await(zerosOrAgain);
  ASSERT_FALSE(zerosOrAgain.isError())
      << zerosOrAgain.getDiagnostic().getMessage() << '\n';
  EXPECT_TRUE(zerosOrAgain->has_value());

  ASSERT_TRUE(readBuf->getBufferSize() == 1)
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
  auto tempDirectory = tempDir.getPath() / "ModularOldVersionString";

  std::error_code ec;
  std::filesystem::create_directory(tempDirectory, ec);
  ASSERT_FALSE(ec) << "failed to create directory: " << ec.message() << "\n";

  // Upon creating a new cache, all of the old versions on the filesystem
  // should be deleted.
  auto fsCache = RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(*runtime, tempDir.getPath()).takeValue());
  ASSERT_TRUE(!std::filesystem::exists(tempDirectory, ec))
      << "expected the temp directory to be deleted by cacheDir creation\n";
}

//===----------------------------------------------------------------------===//
// Specialized FilesystemBackend tests
//===----------------------------------------------------------------------===//

/// Returns key for given thread and run.
static std::string makeKeyStr(int thread, int run) {
  return "key[" + std::to_string(thread) + "," + std::to_string(run) + "]";
}

/// Returns key in buffer form for thread and run.
static BufferRef makeKey(int thread, int run) {
  std::string key = makeKeyStr(thread, run);
  auto writeableKeyBuffer = WriteableBuffer::get();
  writeableKeyBuffer->write(key.data(), key.size());
  return std::move(writeableKeyBuffer);
}

/// Returns buffer with distinguished byte value for thread and run.
static BufferRef makeValue(size_t size, int numThreads, int thread, int run) {
  uint8_t value = (thread * numThreads) + run;
  auto writeableValueBuffer = WriteableBuffer::get(size);
  memset(writeableValueBuffer->getBufferStart(), value, size);
  return std::move(writeableValueBuffer);
}

static LLCL::EncodedLocation unknownLoc() {
  return LLCL::UnknownLocationDecoder::getEncodedLocation();
}

static std::unique_ptr<Runtime> makeRuntime() {
  return LLCL::createUniqueRuntime(
      LLCL::RuntimeOptions().withLeakCheckedAllocator());
}

TEST(FilesystemBackend, Hammer) {
  const size_t size = 8000;
  const int numThreads = 20;
  const int numKeys = 200;
  TempDir tempDir = createTempDir();

  std::vector<std::thread> threads;
  for (int thread = 0; thread < numThreads; ++thread) {
    threads.emplace_back([thread, &tempDir]() {
      auto runtime = makeRuntime();
      auto backend = getFilesystemBackend(*runtime, tempDir.getPath());
      auto threadDone = AsyncValueRef<Chain>::allocate(*runtime);

      // Insert known values with known keys.
      std::vector<AnyAsyncValueRef> insertsDone;
      for (int run = 0; run < numKeys; ++run) {
        insertsDone.emplace_back(backend->insert(
            makeKey(thread, run), makeValue(size, numThreads, thread, run)));
      }
      andThenSyncMoving(insertsDone, [thread, runtime = runtime.get(),
                                      backend = backend.copy(),
                                      threadDone = threadDone.copy()](
                                         MutableArrayRef<AnyAsyncValueRef>
                                             insertsDone) mutable {
        for (auto &ref : insertsDone) {
          if (ref.isError())
            return std::move(threadDone).setToError(ref.takeDiagnostic());
        }

        // Retrieve those values and check they match.
        std::vector<AnyAsyncValueRef> findsDone;
        for (int run = 0; run < numKeys; ++run) {
          auto findDone = AsyncValueRef<Chain>::allocate(*runtime);
          backend->find(makeKey(thread, run))
              .andThenSync([thread, run, findDone = findDone.copy()](
                               AsyncValueRef<std::optional<BufferRef>>
                                   optResult) mutable {
                if (optResult.isError())
                  return std::move(findDone).setToError(
                      optResult.takeDiagnostic());
                if (!optResult->has_value())
                  return std::move(findDone).setToError(
                      {Twine("no entry for ") + makeKeyStr(thread, run),
                       unknownLoc()});
                if (optResult->value()->getBufferSize() != size)
                  return std::move(findDone).setToError(
                      {Twine("mismatched size for ") + makeKeyStr(thread, run) +
                           ": actual size is " +
                           Twine(optResult->value()->getBufferSize()),
                       unknownLoc()});
                BufferRef expectedValue =
                    makeValue(size, numThreads, thread, run);
                if (memcmp(optResult->value()->getBufferStart(),
                           expectedValue->getBufferStart(), size))
                  return std::move(findDone).setToError(
                      {Twine("retrieved value does not match expected value "
                             "for ") +
                           makeKeyStr(thread, run),
                       unknownLoc()});
                std::move(findDone).emplace();
              });
          findsDone.emplace_back(std::move(findDone));
        }
        andThenSyncMoving(
            findsDone,
            [threadDone = std::move(threadDone)](
                MutableArrayRef<AnyAsyncValueRef> findsDone) mutable {
              for (auto &ref : findsDone) {
                if (ref.isError())
                  return std::move(threadDone).setToError(ref.takeDiagnostic());
              }
              std::move(threadDone).emplace();
            });
      });

      await(threadDone);
      EXPECT_FALSE(threadDone.isError())
          << threadDone.getDiagnostic().getMessage().get() << '\n';
    });
  }

  for (auto &thread : threads)
    thread.join();
}
