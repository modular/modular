//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::Mojo::LSP;

TEST(SemanticTokensTest, testSemanticTokens) {
  Document doc("test:///foo.mojo", R"(
import std.builtin
comptime builtin_alias = std.builtin

struct Struct:
  var field: Int

comptime struct_alias = Struct

# `raises` is load-bearing; see MOTO-903.
def foo() raises:
  return

comptime int_alias = 10

trait ATrait:
  def foo(var self, i: Self):
     ...

struct StructWithTrait(ATrait):
    def foo(var self, i: Self):
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

            EXPECT_TRUE(llvm::all_of(tokens, [&](const SemanticToken &token) {
              return token.range.start.line == token.range.end.line &&
                     token.range.start.character <= token.range.end.character;
            }));
          })
      .execute();
}

TEST(SemanticTokensTest, testAddressSpaceModifiers) {
  Document doc("test:///foo.mojo", R"(
from std.collections import Array
from std.memory import AddressSpace

comptime SharedPtr = UnsafePointer[
    Float32, MutAnyOrigin, address_space=AddressSpace.SHARED
]

struct Tile:
    var field_ptr: SharedPtr
    var field_int: Int

    def use(self):
        _ = self.field_ptr
        _ = self.field_int

def kernel[param_ptr: SharedPtr](
    smem: SharedPtr,
    gmem: UnsafePointer[Float32, MutAnyOrigin],
    ref [_, AddressSpace.LOCAL] local_val: Float32,
    plain: Int,
    nested: Array[SharedPtr, 2],
):
    var smem_copy = smem
    _ = smem_copy
    _ = gmem
    _ = local_val
    _ = plain
    _ = param_ptr
    _ = nested
)");

  // Each token carries at most the one modifier bit for its address space.
  auto addressSpaceBit = [](unsigned addressSpace) {
    return 1u << addressSpace;
  };

  createTestClient()
      .open(doc)
      .semanticTokensFull(
          doc,
          [&](ArrayRef<SemanticToken> tokens) {
            auto findToken = [&](llvm::lsp::Range range) {
              return llvm::find_if(tokens, [&](const SemanticToken &token) {
                return token.range == range;
              });
            };

            // The address space parameter of the argument's type (SHARED).
            auto smem = findToken(*doc.findFirstRange("smem"));
            ASSERT_NE(smem, tokens.end());
            EXPECT_EQ(smem->modifiers, addressSpaceBit(3));

            // Variables inherit it from their inferred type, including at
            // their uses.
            auto smemCopy = findToken(*doc.findLastRange("smem_copy"));
            ASSERT_NE(smemCopy, tokens.end());
            EXPECT_EQ(smemCopy->modifiers, addressSpaceBit(3));

            // A generic address space parameter is reported as address space
            // 0.
            auto gmem = findToken(*doc.findFirstRange("gmem"));
            ASSERT_NE(gmem, tokens.end());
            EXPECT_EQ(gmem->modifiers, addressSpaceBit(0));

            // Values without an address space parameter aren't marked, even
            // though they live in generic memory.
            auto plain = findToken(*doc.findFirstRange("plain"));
            ASSERT_NE(plain, tokens.end());
            EXPECT_EQ(plain->modifiers, 0u);

            // An address space nested in a type parameter, here the element
            // type of an array.
            auto nested = findToken(*doc.findFirstRange("nested"));
            ASSERT_NE(nested, tokens.end());
            EXPECT_EQ(nested->modifiers, addressSpaceBit(3));

            // The address space of the reference itself (LOCAL).
            auto localVal = findToken(*doc.findFirstRange("local_val"));
            ASSERT_NE(localVal, tokens.end());
            EXPECT_EQ(localVal->modifiers, addressSpaceBit(5));

            // Struct fields, at their declaration and their uses.
            auto fieldPtr = findToken(*doc.findFirstRange("field_ptr"));
            ASSERT_NE(fieldPtr, tokens.end());
            EXPECT_EQ(fieldPtr->modifiers, addressSpaceBit(3));
            auto fieldPtrUse = findToken(*doc.findLastRange("field_ptr"));
            ASSERT_NE(fieldPtrUse, tokens.end());
            EXPECT_EQ(fieldPtrUse->modifiers, addressSpaceBit(3));
            auto fieldInt = findToken(*doc.findFirstRange("field_int"));
            ASSERT_NE(fieldInt, tokens.end());
            EXPECT_EQ(fieldInt->modifiers, 0u);

            // Parameters, at their declaration and their uses.
            auto paramPtr = findToken(*doc.findFirstRange("param_ptr"));
            ASSERT_NE(paramPtr, tokens.end());
            EXPECT_EQ(paramPtr->modifiers, addressSpaceBit(3));
            auto paramPtrUse = findToken(*doc.findLastRange("param_ptr"));
            ASSERT_NE(paramPtrUse, tokens.end());
            EXPECT_EQ(paramPtrUse->modifiers, addressSpaceBit(3));
          })
      .execute();
}
