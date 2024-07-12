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
