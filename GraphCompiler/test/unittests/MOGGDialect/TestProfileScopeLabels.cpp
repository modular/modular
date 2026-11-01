//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Unit tests for ProfileScopeLabels.h. These exercise edge cases that are
// hard to express in textual MLIR, including ProfileScopeLocationAttr wrapping
// a FusedLoc (the shape AttachPythonStackTraces produces when source
// tracebacks are enabled under an active profile_scope).
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace M::MOGG;

namespace {

class ProfileScopeLabelsTest : public ::testing::Test {
protected:
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};

  void SetUp() override { ctx.loadDialect<MOGGDialect>(); }

  mlir::Location unknown() { return mlir::UnknownLoc::get(&ctx); }

  mlir::Location profileScope(llvm::StringRef name, mlir::Location child,
                              llvm::StringRef color = {}) {
    auto nameAttr = mlir::StringAttr::get(&ctx, name);
    auto colorAttr =
        color.empty() ? mlir::StringAttr{} : mlir::StringAttr::get(&ctx, color);
    return ProfileScopeLocationAttr::get(&ctx, nameAttr, child, colorAttr);
  }

  mlir::Location fused(llvm::ArrayRef<mlir::Location> locs,
                       mlir::Attribute metadata = {}) {
    if (!metadata)
      metadata = mlir::UnitAttr::get(&ctx);
    return mlir::FusedLoc::get(&ctx, locs, metadata);
  }
};

TEST_F(ProfileScopeLabelsTest, FusedLocAtRootProducesPeerLayers) {
  // fused[profile_scope("a"), profile_scope("b")] -> two peer layers.
  auto loc =
      fused({profileScope("a", unknown()), profileScope("b", unknown())});
  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 2u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"a"}));
  EXPECT_EQ(layers[1], (ProfileScopeLayer{"b"}));
}

TEST_F(ProfileScopeLabelsTest, NestedScopesProduceOneLayer) {
  // Python semantics: the last-entered scope is the outermost layer of the
  // location attr, so `with profile_scope("outer"): with
  // profile_scope("inner")` produces profile_scope("inner",
  // profile_scope("outer", leaf)).
  auto loc = profileScope("inner", profileScope("outer", unknown()));
  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 1u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"outer", "inner"}));
}

TEST_F(ProfileScopeLabelsTest, FusedLocInsideProfileScopePreservesOuterLayer) {
  // This is the shape produced by AttachPythonStackTraces when source
  // tracebacks are enabled: the OpaqueLoc leaf inside a profile_scope is
  // rewritten to a FusedLoc. The outer scope must not be lost.
  auto fileLoc = mlir::FileLineColLoc::get(&ctx, "model.py", 10, 0);
  auto traceLoc = fused({fileLoc}, mlir::StringAttr::get(&ctx, "traceback"));
  auto loc = profileScope("draft_forward", traceLoc);

  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 1u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"draft_forward"}));
}

TEST_F(ProfileScopeLabelsTest, FusedLocInsideNestedProfileScopePrefixesPeers) {
  // `with profile_scope("outer"): with profile_scope(...):` => the inner
  // profile_scope layers wrap the outer one, so "outer" is deepest.
  auto inner =
      fused({profileScope("a", unknown()), profileScope("b", unknown())});
  auto loc = profileScope("inner", profileScope("outer", inner));

  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 2u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"outer", "inner", "a"}));
  EXPECT_EQ(layers[1], (ProfileScopeLayer{"outer", "inner", "b"}));
}

TEST_F(ProfileScopeLabelsTest, PeerOrderIsCanonicalized) {
  // fused[profile_scope("b"), profile_scope("a")] -> layers sorted.
  auto loc =
      fused({profileScope("b", unknown()), profileScope("a", unknown())});
  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 2u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"a"}));
  EXPECT_EQ(layers[1], (ProfileScopeLayer{"b"}));
}

TEST_F(ProfileScopeLabelsTest, CallSiteLocAtRootProducesPeerLayers) {
  auto loc = mlir::CallSiteLoc::get(profileScope("callee", unknown()),
                                    profileScope("caller", unknown()));
  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 2u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"callee"}));
  EXPECT_EQ(layers[1], (ProfileScopeLayer{"caller"}));
}

TEST_F(ProfileScopeLabelsTest,
       CallSiteLocInsideProfileScopePreservesOuterLayer) {
  // With an active profile_scope, the accumulated scope prefixes the leaf
  // information inside a CallSiteLoc.
  auto fileLoc = mlir::FileLineColLoc::get(&ctx, "model.py", 10, 0);
  auto callLoc = mlir::CallSiteLoc::get(fileLoc, unknown());
  auto loc = profileScope("inner", profileScope("outer", callLoc));

  auto layers = collectProfileScopeLayers(loc);
  ASSERT_EQ(layers.size(), 1u);
  EXPECT_EQ(layers[0], (ProfileScopeLayer{"outer", "inner"}));
}

TEST_F(ProfileScopeLabelsTest, FormatPrefixAndSuffixDifferOnlyInBrackets) {
  llvm::SmallVector<ProfileScopeLayer, 2> layers = {
      ProfileScopeLayer{"draft_forward", "inner"},
      ProfileScopeLayer{"target_forward"}};

  EXPECT_EQ(formatProfileScope(layers, true),
            "[draft_forward/inner:target_forward] ");
  EXPECT_EQ(formatProfileScope(layers, false),
            " [draft_forward/inner:target_forward]");
}

} // namespace
