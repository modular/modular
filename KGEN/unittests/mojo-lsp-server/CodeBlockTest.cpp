//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(CodeBlockTest, testCodeBlockDiagnostics) {
  Document doc("test:///foo.mojo",
               R"(
fn function():
  """Test doc string.

  ```mojo
  var foo = bar
  ```
  """
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "use of unknown declaration 'bar'");
                     })
      .execute();
}

TEST(CodeBlockTest, testCodeBlockHover) {
  Document doc("test:///foo.mojo",
               R"(
fn function():
  """Test doc string.

  ```mojo
  fn test():
    var foo: Int = 420
    var bar = 1 + `foo`
    print(bar)
  ```

  """
)");

  createTestClient()
      .open(doc)
      .hover(doc, *doc.findFirstPos("foo"),
             [](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(variable) var foo: Int
```)");
               EXPECT_EQ(hover.range, lsp::Range({6, 8}, {6, 11}));
             })
      .execute();
}

TEST(CodeBlockTest, testCodeBlockCompletion) {
  Document doc("test:///foo.mojo",
               R"(
fn function():
  """Test doc string.

  ```mojo
  var foo = 10
  ```

  ```mojo
  foo.completion
  ```

  """
)");

  createTestClient()
      .open(doc)
      .completion(doc, *doc.findFirstPos("completion"),
                  [](const lsp::CompletionList &completion) {
                    EXPECT_TRUE(llvm::any_of(
                        completion.items, [](const lsp::CompletionItem &item) {
                          return item.label == "value" &&
                                 item.kind == lsp::CompletionItemKind::Field;
                        }));
                  })
      .execute();
}

TEST(CodeBlockTest, testCodeBlockEndCompletion) {
  Document doc = createDocumentFromInputFileWithinPackage("doc_strings.mojo");

  createTestClient()
      .open(doc)
      .completion(doc, doc.findFirstRange("test_completions.")->end,
                  [](const lsp::CompletionList &completion) {
                    EXPECT_TRUE(llvm::any_of(
                        completion.items, [](const lsp::CompletionItem &item) {
                          return item.label == "completion_test" &&
                                 item.kind == lsp::CompletionItemKind::Function;
                        }));
                  })
      .execute();
}
