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

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace testing;

namespace {

/// Drives the ForceAlwaysInline pass over a module built by hand, so a test can
/// assert both what the `onAnnotated` callback sees and the inline level the
/// pass leaves behind.
class ForceAlwaysInlineTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  OpBuilder builder{&ctx};
  OwningOpRef<ModuleOp> module;

  ForceAlwaysInlineTest() {
    ctx.loadDialect<KGENDialect>();
    ctx.loadDialect<HLCF::HLCFDialect>();
    module = ModuleOp::create(builder, builder.getUnknownLoc());
  }

  /// Inserts a `kgen.func` named `name` at the end of the module.
  FuncOp addFunc(StringRef name, InlineLevel level, bool external = false) {
    builder.setInsertionPointToEnd(module->getBody());
    FuncOp func = FuncOp::create(
        builder, builder.getUnknownLoc(), builder.getStringAttr(name),
        FuncType::get(&ctx, {}, {}), level, ExportKind::NotExported, external);
    if (!external) {
      builder.setInsertionPointToStart(
          builder.createBlock(&func.getBodyRegion()));
      HLCF::ReturnOp::create(builder, builder.getUnknownLoc());
    }
    return func;
  }

  /// Runs the pass with `onAnnotated` and returns the warnings it emitted.
  std::vector<std::string> run(const std::function<void(FuncOp)> &onAnnotated) {
    std::vector<std::string> warnings;
    auto recordWarning = [&](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Warning)
        warnings.push_back(diag.str());
    };
    ScopedDiagnosticHandler handler(&ctx, recordWarning);
    PassManager pm(&ctx);
    pm.addNestedPass<FuncOp>(createForceAlwaysInline(onAnnotated));
    EXPECT_TRUE(succeeded(pm.run(*module)));
    return warnings;
  }
};

/// The callback is how a driver turns an overridden `@no_inline` into a
/// user-visible diagnostic. It must fire for that function and for no other.
TEST_F(ForceAlwaysInlineTest, WarnsWhenNoInlineIsOverridden) {
  FuncOp noInline = addFunc("no_inline_fn", InlineLevel::Never);
  addFunc("plain", InlineLevel::Automatic);

  std::vector<std::string> annotated;
  std::vector<std::string> warnings = run([&](FuncOp func) {
    annotated.push_back(func.getSymName().str());
    if (inlineLevelOrAutomatic(func.getInlineLevel()) == InlineLevel::Never)
      func.emitWarning() << "overriding no_inline: function is force-inlined";
  });

  ASSERT_EQ(annotated.size(), 1u);
  EXPECT_EQ(annotated.front(), "no_inline_fn");
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings.front().find("overriding no_inline"), std::string::npos);
  EXPECT_EQ(inlineLevelOrAutomatic(noInline.getInlineLevel()),
            InlineLevel::Always);
}

/// Automatic functions are the pass's normal input, so they never reach the
/// callback. An external function carries no body to inline and is skipped
/// whole, annotation included.
TEST_F(ForceAlwaysInlineTest, VisitsEveryNonAutomaticLevelButNotExternal) {
  addFunc("plain", InlineLevel::Automatic);
  FuncOp always = addFunc("always_fn", InlineLevel::Always);
  FuncOp never = addFunc("no_inline_fn", InlineLevel::Never);
  FuncOp external =
      addFunc("external_fn", InlineLevel::Never, /*external=*/true);

  std::vector<std::string> annotated;
  std::vector<std::string> warnings =
      run([&](FuncOp func) { annotated.push_back(func.getSymName().str()); });

  EXPECT_EQ(annotated, (std::vector<std::string>{"always_fn", "no_inline_fn"}));
  EXPECT_TRUE(warnings.empty());
  EXPECT_EQ(inlineLevelOrAutomatic(always.getInlineLevel()),
            InlineLevel::Always);
  EXPECT_EQ(inlineLevelOrAutomatic(never.getInlineLevel()),
            InlineLevel::Always);
  EXPECT_EQ(inlineLevelOrAutomatic(external.getInlineLevel()),
            InlineLevel::Never);
}

/// A driver that has nothing to report passes no callback at all.
TEST_F(ForceAlwaysInlineTest, CallbackIsOptional) {
  FuncOp func = addFunc("no_inline_fn", InlineLevel::Never);

  EXPECT_TRUE(run(/*onAnnotated=*/{}).empty());
  EXPECT_EQ(inlineLevelOrAutomatic(func.getInlineLevel()), InlineLevel::Always);
}

} // namespace
