//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/Buffer.h"

#include "gtest/gtest.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

TEST(BufferTest, RefCountingWorks) {
  auto buffer = Buffer::get("hello");
  EXPECT_TRUE(buffer->getNumReferences() == 1);
  // Move the buffer into a new one.
  auto buffer2 = std::move(buffer);
  EXPECT_TRUE(buffer2->getNumReferences() == 1);
  EXPECT_TRUE(buffer2->getBuffer() == "hello");

  {
    auto buffer3 = buffer2.copy();
    // These two should have the exact same data pointers.
    EXPECT_TRUE(buffer2->getBufferStart() == buffer3->getBufferStart());
    // There should now be 2 references.
    EXPECT_TRUE(buffer2->getNumReferences() == 2);
  }
  // And the data should still not have been freed.
  EXPECT_TRUE(buffer2->getBuffer() == "hello");
  EXPECT_TRUE(buffer2->getNumReferences() == 1);
}

TEST(BufferTest, TestWrite) {
  auto buffer = WriteableBuffer::get();
  *buffer << "hello";
  EXPECT_TRUE(buffer->getBuffer() == "hello")
      << "Actually had: " << buffer->getBuffer();

  auto buffer2 = std::move(buffer);
  EXPECT_TRUE(buffer2->getBuffer() == "hello")
      << "Actually had: " << buffer2->getBuffer();

  auto buffer3 = buffer2.copy();
  EXPECT_TRUE(buffer3->getBufferStart() == buffer2->getBufferStart());
}

TEST(BufferTest, TestReadWriteFile) {
  auto writeOr = WriteableBuffer::getFile("tmpFile", /*size=*/5, /*offset=*/0);
  EXPECT_FALSE(writeOr.isError()) << writeOr.getError();
  WriteableBufferRef write = std::move(*writeOr);
  // pwrite because we want to write to a particular offset.
  char hello[] = "hello";
  write->pwrite(hello, 5, 0);

  auto wrongBufferOr = Buffer::getFile("aSillyNamedTempFileThatShouldNotExist");
  EXPECT_TRUE(wrongBufferOr.isError()) << "No such file should exist...";

  auto rightBufferOr = Buffer::getFile("tmpFile");
  EXPECT_FALSE(rightBufferOr.isError()) << rightBufferOr.getError();
  EXPECT_TRUE((*rightBufferOr)->getBuffer() == "hello");

  // Clean up the file.
  llvm::sys::fs::remove("tmpFile");
}
