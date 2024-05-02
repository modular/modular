//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::Mojo::LSP;

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

  createTestClient()
      .open(doc)
      .semanticTokensFull(
          doc,
          [&](ArrayRef<SemanticToken> tokens) {
            EXPECT_NE((int)tokens.size(), 0);
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("builtin") &&
                     token.kind == SemanticTokenKind::kModule;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("builtin_alias") &&
                     token.kind == SemanticTokenKind::kModule;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("Struct") &&
                     token.kind == SemanticTokenKind::kClass;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("struct_alias") &&
                     token.kind == SemanticTokenKind::kType;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("field") &&
                     token.kind == SemanticTokenKind::kField;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("foo") &&
                     token.kind == SemanticTokenKind::kFunction;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("int_alias") &&
                     token.kind == SemanticTokenKind::kVariable;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("ATrait") &&
                     token.kind == SemanticTokenKind::kTrait;
            }));
            EXPECT_TRUE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range == *doc.findFirstRange("Self") &&
                     token.kind == SemanticTokenKind::kTrait;
            }));
            // Check that we didn't add a token for the synthetic methods of the
            // StructWithTrait struct.
            EXPECT_FALSE(llvm::any_of(tokens, [&](const SemanticToken &token) {
              return token.range ==
                         *doc.findLastPos("struct StructWithTrait") &&
                     token.kind == SemanticTokenKind::kFunction;
            }));
          })
      .execute();
}
