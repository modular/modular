//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ErrorOr.h"
#include "Support/Error.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"

#include "gtest/gtest.h"

using namespace M;

// TODO(akirchhoff): These tests are fairly minimal.  More would be better.

TEST(ErrorOr, successful) {
  ErrorOr<int> eo(5);
  EXPECT_FALSE(eo.isError());
  EXPECT_FALSE(eo);
  EXPECT_TRUE(LogicalResult(eo).succeeded());
  EXPECT_EQ(5, eo.get());
  EXPECT_EQ(5, *eo);
  EXPECT_EQ(nullptr, eo.getError());
}

TEST(ErrorOr, erroneous) {
  ErrorOr<int> eo(Error("Toaster is broken"));
  EXPECT_TRUE(eo.isError());
  EXPECT_TRUE(eo);
  EXPECT_FALSE(LogicalResult(eo).succeeded());
  EXPECT_STREQ("Toaster is broken", eo.getError());
}

TEST(ErrorOr, erroneousTwine) {
  ErrorOr<int> eo(Error(llvm::Twine("Toaster is broken")));
  EXPECT_TRUE(eo.isError());
  EXPECT_TRUE(eo);
  EXPECT_FALSE(LogicalResult(eo).succeeded());
  EXPECT_STREQ("Toaster is broken", eo.getError());
}

TEST(ErrorOr, equality) {
  EXPECT_EQ(ErrorOr<int>(5), ErrorOr<int>(5));
  EXPECT_NE(ErrorOr<int>(5), ErrorOr<int>(10));
  EXPECT_EQ(ErrorOr<int>(Error("File not found")),
            ErrorOr<int>(Error("File not found")));
  EXPECT_NE(ErrorOr<int>(Error("File not found")),
            ErrorOr<int>(Error("Network connection lost")));
  EXPECT_NE(ErrorOr<int>(5), ErrorOr<int>(Error("File not found")));
}

// TODO(akirchhoff): Test move semantics
// TODO(akirchhoff): Test copying
