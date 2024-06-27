//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(DiagnosticsTest, testDiagnosticsInvalidImport) {
  Document doc("test:///foo.mojo",
               R"(
from a.b.c import d
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 2);
                       EXPECT_EQ(diags[0].message,
                                 "unable to locate module 'a'");
                     })
      .execute();
}

TEST(DiagnosticsTest, detectUnusedLocalVariable) {
  Document doc("test:///unused.mojo", R"(
fn function():
  var unused = 0
  var used = 1
  print(used)
  )");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message, "unused variable 'unused'");
                     })
      .execute();
}

TEST(DiagnosticsTest, detectUnusedWithVar) {
  Document doc("test:///unused.mojo", R"(
fn function() raises:
  with open("file.txt", "r") as file:
    pass
  )");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message, "unused variable 'file'");
                     })
      .execute();
}

TEST(DiagnosticsTest, detectUnusedForVar) {
  Document doc("test:///unused.mojo", R"(
fn function():
  for x in range(5):
    pass
  )");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message, "unused variable 'x'");
                     })
      .execute();
}

TEST(DiagnosticsTest, ignoreUnusedWithUnderscore) {
  Document doc("test:///unused.mojo", R"(
fn function() raises:
  var _unused_var = 0

  with open("file.txt", "r") as _file:
    pass

  for _x in range(5):
    pass
  )");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 0);
                     })
      .execute();
}
