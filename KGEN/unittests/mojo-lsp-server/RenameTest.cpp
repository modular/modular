//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(RenameTest, testRename) {
  Document doc("test:///foo.mojo", R"(
fn main():
  var someVar = 123
  print(someVar)
  )");

  createTestClient()
      .open(doc)
      .rename(doc, *doc.findFirstPos("someVar"), "abc",
              [&doc](const lsp::WorkspaceEdit &edit) {
                ASSERT_EQ(edit.changes.size(), (size_t)1);
                ASSERT_EQ(edit.changes.count("/foo.mojo"), (size_t)1);

                const std::vector<lsp::TextEdit> &changes =
                    edit.changes.at("/foo.mojo");

                ASSERT_EQ(changes.size(), (size_t)2);

                std::vector<lsp::Range> ranges = doc.findAllRanges("someVar");
                ASSERT_EQ(ranges.size(), (size_t)2);

                EXPECT_EQ(changes.at(0).range, ranges.at(0));
                EXPECT_EQ(changes.at(0).newText, "abc");

                EXPECT_EQ(changes.at(1).range, ranges.at(1));
                EXPECT_EQ(changes.at(1).newText, "abc");
              })
      .execute();
}

TEST(RenameTest, cannotRenameGlobal) {
  Document doc("test:///foo.mojo", R"(
var someGlobal = 123
  )");

  createTestClient()
      .open(doc)
      .renameError(doc, *doc.findFirstPos("someGlobal"), "newGlobal",
                   [](const lsp::LSPError2 &error) {
                     EXPECT_EQ(
                         error.message,
                         "renaming is only available for local variables");
                     EXPECT_EQ(error.code, lsp::ErrorCode::InvalidRequest);
                   })
      .execute();
}

TEST(RenameTest, cannotRenameArgument) {
  Document doc("test:///foo.mojo", R"(
fn something(arg: Int):
  pass
  )");

  createTestClient()
      .open(doc)
      .renameError(doc, *doc.findFirstPos("arg"), "argument",
                   [](const lsp::LSPError2 &error) {
                     EXPECT_EQ(
                         error.message,
                         "renaming is only available for local variables");
                     EXPECT_EQ(error.code, lsp::ErrorCode::InvalidRequest);
                   })
      .execute();
}

TEST(RenameTest, cannotRenameExternalSymbol) {
  Document doc("test:///foo.mojo", R"(
fn main():
  print(1 + 2)
  )");

  createTestClient()
      .open(doc)
      .renameError(doc, *doc.findFirstPos("print"), "myPrint",
                   [](const lsp::LSPError2 &error) {
                     EXPECT_EQ(
                         error.message,
                         "renaming is only available for local variables");
                     EXPECT_EQ(error.code, lsp::ErrorCode::InvalidRequest);
                   })

      .execute();
}

TEST(RenameTest, cannotRenameNonSymbol) {
  Document doc("test:///foo.mojo", R"(

fn main():
  print(1 + 2)
  )");

  createTestClient()
      .open(doc)
      .renameError(doc, lsp::Position(0, 0), "asdf",
                   [](const lsp::LSPError2 &error) {
                     EXPECT_EQ(error.message,
                               "no identified symbol at this position");
                     EXPECT_EQ(error.code, lsp::ErrorCode::InvalidRequest);
                   })

      .execute();
}
