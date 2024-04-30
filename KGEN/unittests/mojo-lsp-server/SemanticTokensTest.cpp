//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(SemanticTokensTest, testSemanticTokens) {
  Document doc("test:///foo.mojo", R"(
import builtin
alias builtin_alias = builtin

struct Struct:
  var field: Int

alias struct_alias = Struct

fn foo():
  return

alias int_alias = 10

trait ATrait:
  fn foo(owned self, i: Self):
     ...

struct StructWithTrait(ATrait):
    fn foo(owned self, i: Self):
        pass
)");

  /* Commented out just to prevent compilation warnings.
  const auto kVariable = 0;
  const auto kSpecialVariable = 1;
  const auto kParameter = 2;
  const auto kFunction = 3;
  const auto kMethod = 4;
  const auto kProperty = 5;
  const auto kClass = 6;
  const auto kInterface = 7;
  const auto kType = 8;
  const auto kNamespace = 9;
  */

  createTestClient()
      .open(doc)
      // FIXME(MOTO-391): This call crashes the parser.
      /*
      .semanticTokensFull(
          doc,
          [&](const lsp::SemanticTokens &tokens) {
            EXPECT_NE((int)tokens.tokens.size(), 0);
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("builtin") &&
                         token.tokenType == kNamespace;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("builtin_alias") &&
                         token.tokenType == kNamespace;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("Struct") &&
                         token.tokenType == kClass;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("struct_alias") &&
                         token.tokenType == kType;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("field") &&
                         token.tokenType == kProperty;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("foo") &&
                         token.tokenType == kFunction;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("int_alias") &&
                         token.tokenType == kVariable;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("ATrait") &&
                         token.tokenType == kInterface;
                }));
            EXPECT_TRUE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return lsp::Range(token.deltaLine, token.deltaStart) ==
                             *doc.findFirstRange("Self") &&
                         token.tokenType == kInterface;
                }));
            // Check that we didn't add a token for the synthetic methods of the
            // StructWithTrait struct.
            EXPECT_FALSE(llvm::any_of(
                tokens.tokens, [&](const lsp::SemanticToken &token) {
                  return token.deltaLine ==
                             *doc.findLastPos("struct StructWithTrait") &&
                         token.tokenType == kFunction;
                }));
          })
          */
      .execute();
}
