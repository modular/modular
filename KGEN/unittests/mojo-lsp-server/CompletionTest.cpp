//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(CompletionTest, testCompletionImport) {
  Document doc("test:///foo.mojo",
               R"(
import b

# This is a comment.
)");

  createTestClient()
      .open(doc)
      .completion(
          doc, lsp::Position{1, 8},
          [](const lsp::CompletionList &completionList) {
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "builtin" &&
                         item.kind == lsp::CompletionItemKind::Folder &&
                         item.documentation &&
                         StringRef(item.documentation->value)
                             .contains("Implements the builtin package");
                }));
          })
      .execute();
}

TEST(CompletionTest, testCompletionNestedImport) {
  Document doc("test:///foo.mojo",
               R"(
import builtin.
)");

  createTestClient()
      .open(doc)
      .completion(
          doc, lsp::Position{1, 15},
          [](const lsp::CompletionList &completionList) {
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "bool" &&
                         item.kind == lsp::CompletionItemKind::Module;
                }));
          })
      .execute();
}

TEST(CompletionTest, testCompletionRelativeImport) {
  Document doc = createDocumentFromInputFile("imports.mojo");

  createTestClient()
      .open(doc)
      .completion(
          doc, doc.findFirstRange("from .aliases")->end,
          [](const lsp::CompletionList &completionList) {
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "aliases" &&
                         item.kind == lsp::CompletionItemKind::Module;
                }));
          })
      .execute();
}

TEST(CompletionTest, testCompletionImportMember) {
  Document doc("test:///foo.mojo",
               R"(
from memory.unsafe import D
)");

  createTestClient()
      .open(doc)
      .completion(
          doc, lsp::Position(1, 27),
          [](const lsp::CompletionList &completionList) {
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "DTypePointer" &&
                         item.kind == lsp::CompletionItemKind::Struct;
                }));
          })
      .execute();
}

TEST(CompletionTest, testCompletionItemSorting) {
  Document doc("test:///foo.mojo",
               R"(
@value
struct Foo:
  var __other: Int
  var ___another__: Int
  var __dunder__: Int
  var _sunder_: Int
  var _priv: Int
  var normal: Int
  fn foo(self): pass
  fn _foobar_(self): pass
  fn _bar(self): pass
  fn __baz__(self): pass

fn function(arg: Foo):
  arg.
)");

  createTestClient()
      .open(doc)
      .completion(
          doc, lsp::Position(15, 6),
          [](const lsp::CompletionList &completionList) {
            EXPECT_STREQ(completionList.items[0].label.c_str(), "foo");
            EXPECT_STREQ(completionList.items[1].label.c_str(), "normal");
            EXPECT_STREQ(completionList.items[2].label.c_str(), "_bar");
            EXPECT_STREQ(completionList.items[3].label.c_str(), "_priv");
            EXPECT_STREQ(completionList.items[4].label.c_str(), "_foobar_");
            EXPECT_STREQ(completionList.items[5].label.c_str(), "_sunder_");
            EXPECT_STREQ(completionList.items[6].label.c_str(), "__baz__");
            EXPECT_STREQ(completionList.items[7].label.c_str(), "__copyinit__");
            EXPECT_STREQ(completionList.items[8].label.c_str(), "__del__");
            EXPECT_STREQ(completionList.items[9].label.c_str(), "__init__");
            EXPECT_STREQ(completionList.items[10].label.c_str(),
                         "__moveinit__");
            EXPECT_STREQ(completionList.items[11].label.c_str(), "__dunder__");
            EXPECT_STREQ(completionList.items[12].label.c_str(),
                         "___another__");
            EXPECT_STREQ(completionList.items[13].label.c_str(), "__other");
          })
      .execute();
}
TEST(CompletionTest, testCompletionMemberLookup) {
  Document doc("test:///foo.mojo",
               R"(
fn function(arg: Int):
    arg.
)");

  createTestClient()
      .open(doc)
      .completion(
          doc, lsp::Position(2, 8),
          [](const lsp::CompletionList &completionList) {
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "__add__" &&
                         item.kind == lsp::CompletionItemKind::Function;
                }));
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "value" &&
                         item.kind == lsp::CompletionItemKind::Field;
                }));
          })
      .execute();
}

TEST(CompletionTest, testCompletionTopLevelLookup) {
  Document doc("test:///foo.mojo",
               R"(
fn function() -> Int:
    var value: Int = 10
    return value
)");

  createTestClient()
      .open(doc)
      .completion(
          doc, *doc.findFirstPos("nt"),
          [](const lsp::CompletionList &completionList) {
            // Check that we can complete the `Int` from `I` in the result type.
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "Int" &&
                         item.kind == lsp::CompletionItemKind::Struct;
                }));
          })
      .completion(
          doc, *doc.findLastPos("alue"),
          [](const lsp::CompletionList &completionList) {
            // Check that we can complete the `value` from `v` in the return
            // statement.
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "value" &&
                         item.kind == lsp::CompletionItemKind::Variable;
                }));
          })
      .execute();
}

void checkPartialCompoundStatement(const Document &doc, StringRef completeAt) {
  createTestClient()
      .open(doc)
      .completion(doc, *doc.findFirstPos(completeAt),
                  [](const lsp::CompletionList &completionList) {
                    EXPECT_FALSE(completionList.items.empty());
                  })
      .execute();
}

TEST(CompletionTest, testCompletionPartialFn) {
  Document doc("test:///fn_no_colon.mojo",
               R"(
fn function(arg: Int)
)");
  checkPartialCompoundStatement(doc, "nt");
}

TEST(CompletionTest, testCompletionPartialIf) {
  Document doc("test:///if_no_colon.mojo",
               R"(
fn function(arg: Int):
    if arg.value
)");
  checkPartialCompoundStatement(doc, "value");
}

TEST(CompletionTest, testCompletionPartialElif) {
  Document doc("test:///elif_no_colon.mojo",
               R"(
fn function(arg: Int):
    if False:
        return
    elif arg.value
)");
  checkPartialCompoundStatement(doc, "value");
}

TEST(CompletionTest, testCompletionPartialWhile) {
  Document doc("test:///while_no_colon.mojo",
               R"(
fn function(arg: Int):
    while arg.value
)");
  checkPartialCompoundStatement(doc, "value");
}

TEST(CompletionTest, testCompletionPartialWith) {
  Document doc("test:///with_no_colon.mojo",
               R"(
fn function(arg: Int):
    with arg.value
)");
  checkPartialCompoundStatement(doc, "value");
}
