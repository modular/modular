//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Error.h"
#include "llvm/ADT/Twine.h"

#include <utility>

#include "gtest/gtest.h"

using namespace M;

TEST(Error, implicitStaticString) {
  Error error("Toaster is broken");
  EXPECT_STREQ("Toaster is broken", error.get());
}

TEST(Error, implicitStaticStringCopy) {
  Error error("Toaster is broken");
  Error errorCopy = error.copy();
  EXPECT_STREQ("Toaster is broken", errorCopy.get());
}

TEST(Error, implicitStaticStringMoveConstruct) {
  Error error("Toaster is broken");
  Error errorMoved(std::move(error));
  EXPECT_STREQ("Toaster is broken", errorMoved.get());
}

TEST(Error, implicitStaticStringMoveAssign) {
  Error error("Toaster is broken");
  Error errorMoved("Previous value");
  errorMoved = std::move(error);
  EXPECT_STREQ("Toaster is broken", errorMoved.get());
}

TEST(Error, explicitStaticString) {
  Error error = Error::getStaticString("Toaster is broken");
  EXPECT_STREQ("Toaster is broken", error.get());
}

TEST(Error, explicitStaticStringCopy) {
  Error error = Error::getStaticString("Toaster is broken");
  Error errorCopy = error.copy();
  EXPECT_STREQ("Toaster is broken", errorCopy.get());
}

TEST(Error, explicitStaticStringMoveConstruct) {
  Error error = Error::getStaticString("Toaster is broken");
  Error errorMoved(std::move(error));
  EXPECT_STREQ("Toaster is broken", errorMoved.get());
}

TEST(Error, explicitStaticStringMoveAssign) {
  Error error = Error::getStaticString("Toaster is broken");
  Error errorMoved = Error::getStaticString("Previous value");
  errorMoved = std::move(error);
  EXPECT_STREQ("Toaster is broken", errorMoved.get());
}

TEST(Error, twine) {
  Error error(llvm::Twine("All ") + llvm::Twine("eight") +
              llvm::Twine(" toasters are broken"));
  EXPECT_STREQ("All eight toasters are broken", error.get());
}

TEST(Error, twineCopy) {
  Error error(llvm::Twine("All ") + llvm::Twine("eight") +
              llvm::Twine(" toasters are broken"));
  Error errorCopy = error.copy();
  EXPECT_STREQ("All eight toasters are broken", errorCopy.get());
}

TEST(Error, twineMoveConstruct) {
  Error error(llvm::Twine("Toaster is broken"));
  Error errorMoved(std::move(error));
  EXPECT_STREQ("Toaster is broken", errorMoved.get());
}

TEST(Error, twineMoveAssign) {
  Error error(llvm::Twine("Toaster is broken"));
  Error errorMoved(llvm::Twine("Previous value"));
  errorMoved = std::move(error);
  EXPECT_STREQ("Toaster is broken", errorMoved.get());
}
