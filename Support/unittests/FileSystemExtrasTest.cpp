//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/FileSystemExtras.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"
#include <thread>

using namespace M;

/// This test checks that each thread appends to the file serializably - *not*
/// that it must happen in a specific order. The only guarantee provided is that
/// the thread won't be interrupted!
TEST(FileSystemExtras, Append) {
  // Use lots of threads so we definitely have them executing concurrently.
  const int numThreads = 40;
  const int numValues = 32;

  // Create a temp file.
  auto tmpFileOr = TempFile::create("test-%%%%%%%.tmp");
  ASSERT_FALSE(tmpFileOr.isError()) << tmpFileOr.getError();

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  // Each thread will write a list of integers to the temp file.
  for (int thread = 0; thread < numThreads; ++thread) {
    threads.emplace_back([thread, &tmpFileOr]() {
      auto err = appendFileUnderLock(tmpFileOr->getPath(),
                                     [thread](llvm::raw_ostream &os) {
                                       for (int i = 0; i < numValues; ++i)
                                         os << std::to_string(thread) << ",";
                                       os << "\n";
                                     });
      ASSERT_FALSE(err.isError()) << err.getError();
    });
  }

  for (auto &thread : threads)
    thread.join();

  // Open the temp file and read it.
  auto bufOr = llvm::MemoryBuffer::getFile(tmpFileOr->getPath().string(),
                                           /*IsText=*/true);
  ASSERT_TRUE(bufOr) << bufOr.getError().message();
  std::unique_ptr<llvm::MemoryBuffer> buf = std::move(*bufOr);

  // Parse the string - we should have a comma-separated list of 10 integers.
  StringRef buffer = buf->getBuffer();
  std::vector<int> values;
  int lastVal = -1;
  // For each line in the buffer (one per thread), we'll parse 32
  // comma-separated values.
  for (int thread = 0; thread < numThreads; ++thread) {
    int val = 0;
    for (int i = 0; i < numValues; ++i) {
      ASSERT_FALSE(buffer.consumeInteger(10, val));
      // Assert they're all the same.
      if (lastVal != -1)
        ASSERT_EQ(val, lastVal);
      else
        lastVal = val;
      buffer.consume_front(",");
    }
    // Append the value to the vector and reset lastVal.
    values.push_back(val);
    lastVal = -1;
    buffer.consume_front("\n");
  }

  // Just ensure we found every thread number in the vector.
  for (int thread = 0; thread < numThreads; ++thread)
    ASSERT_TRUE(llvm::find(values, thread) != values.end());
}
