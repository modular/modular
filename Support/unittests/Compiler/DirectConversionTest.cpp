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

#include "Support/Compiler/DirectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace M;
using namespace mlir;

namespace {

/// Widens i32 to i64 and leaves every other type unconvertible.
TypeConverter makeWideningConverter() {
  TypeConverter converter;
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (type.isInteger(32))
      return IntegerType::get(type.getContext(), 64);
    if (type.isInteger(64))
      return type;
    return Type();
  });
  return converter;
}

struct WidenConstant : DirectOpConversionPattern<arith::ConstantOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                                DirectRewriter &rewriter) const override {
    auto value = dyn_cast<IntegerAttr>(op.getValue());
    if (!value || !value.getType().isInteger(32))
      return failure();
    Type wide = getTypeConverter()->convertType(value.getType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, IntegerAttr::get(wide, value.getInt()));
    return success();
  }
};

struct WidenAdd : DirectOpConversionPattern<arith::AddIOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;
  LogicalResult matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    // Ops the driver creates are converted too, so never match own output.
    if (!op.getType().isInteger(32))
      return failure();
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

/// Must not run on unconvertible operand types.
struct EraseAddF : DirectOpConversionPattern<arith::AddFOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;
  LogicalResult matchAndRewrite(arith::AddFOp op, OpAdaptor,
                                DirectRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};

class DirectConversionTest : public testing::Test {
protected:
  explicit DirectConversionTest(
      MLIRContext::Threading threading = MLIRContext::Threading::DISABLED)
      : ctx(threading), converter(makeWideningConverter()) {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef text) {
    return parseSourceString<ModuleOp>(text, &ctx);
  }

  FrozenRewritePatternSet makePatterns() {
    RewritePatternSet patterns(&ctx);
    patterns.add<WidenConstant, WidenAdd, EraseAddF>(converter, &ctx);
    return FrozenRewritePatternSet(std::move(patterns));
  }

  /// Runs the driver over `module`, returning the error text if it fails.
  std::optional<std::string>
  convert(ModuleOp module, llvm::function_ref<bool(Operation *)> skipOp = {},
          llvm::function_ref<bool(Operation *)> mustConvert = {}) {
    std::string error;
    ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
      error += diag.str();
      return success();
    });
    if (succeeded(applyDirectConversion(module, &converter, makePatterns(),
                                        skipOp, mustConvert))) {
      return std::nullopt;
    }
    return error;
  }

  static unsigned count(ModuleOp module, StringRef name) {
    unsigned n = 0;
    module.walk(
        [&](Operation *op) { n += op->getName().getStringRef() == name; });
    return n;
  }

  MLIRContext ctx;
  TypeConverter converter;
};

/// Runs the driver's module walks on several threads.
class ThreadedDirectConversionTest : public DirectConversionTest {
protected:
  ThreadedDirectConversionTest()
      : DirectConversionTest(MLIRContext::Threading::ENABLED) {}

  /// Returns a module of `numFuncs` functions. `body` starts on the second
  /// line of each function.
  static std::string makeModuleText(unsigned numFuncs, StringRef body) {
    std::string text;
    for (unsigned i = 0; i < numFuncs; ++i) {
      text +=
          ("func.func @f" + Twine(i) + "(%x: i32) -> i32 {\n" + body + "\n}\n")
              .str();
    }
    return text;
  }
};

} // namespace

TEST_F(DirectConversionTest, RewritesInPlaceAndBridgesUnconvertedUsers) {
  auto module = parse(R"(
    func.func @f() -> i32 {
      %a = arith.constant 1 : i32
      %b = arith.addi %a, %a : i32
      return %b : i32
    })");
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));

  // `return` has no pattern and still wants an i32, so it gets one bridge cast.
  module->walk([](Operation *op) {
    if (isa<arith::ConstantOp, arith::AddIOp>(op)) {
      EXPECT_TRUE(op->getResult(0).getType().isInteger(64));
    }
  });
  EXPECT_EQ(count(*module, "builtin.unrealized_conversion_cast"), 1u);
}

TEST_F(DirectConversionTest, OpsWithoutPatternStayUnconverted) {
  auto module = parse(R"(
    func.func @f(%x: i32) -> i32 {
      %y = arith.muli %x, %x : i32
      return %y : i32
    })");
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));
  EXPECT_EQ(count(*module, "arith.muli"), 1u);
}

TEST_F(DirectConversionTest, MustConvertRejectsOpsNoPatternMatches) {
  auto module = parse(R"(
    func.func @f(%x: i32) -> i32 {
      %y = arith.muli %x, %x : i32
      return %y : i32
    })");
  ASSERT_TRUE(module);
  std::optional<std::string> error = convert(
      *module, {}, [](Operation *op) { return isa<arith::MulIOp>(op); });
  ASSERT_TRUE(error);
  EXPECT_NE(error->find("failed to legalize operation 'arith.muli'"),
            std::string::npos);
}

TEST_F(DirectConversionTest, SkippedOpsAreNeitherConvertedNorRequired) {
  auto module = parse(R"(
    func.func @f() -> i32 {
      %a = arith.constant 1 : i32
      return %a : i32
    })");
  ASSERT_TRUE(module);
  auto skip = [](Operation *op) { return isa<arith::ConstantOp>(op); };
  EXPECT_FALSE(convert(*module, skip, skip));
  EXPECT_EQ(count(*module, "builtin.unrealized_conversion_cast"), 0u);
}

TEST_F(DirectConversionTest, PatternsDoNotRunOnUnconvertibleOperandTypes) {
  // f32 has no conversion, so the addf pattern must not run.
  auto module = parse(R"(
    func.func @f(%x: f32) -> f32 {
      %c = arith.constant 1 : i32
      %y = arith.addf %x, %x : f32
      return %y : f32
    })");
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));
  EXPECT_EQ(count(*module, "arith.addf"), 1u);
}

TEST_F(DirectConversionTest, FoldsACastChainThatReturnsToItsSourceType) {
  auto module = parse(R"(
    func.func @f(%x: i16) -> i16 {
      %a = builtin.unrealized_conversion_cast %x : i16 to i8
      %b = builtin.unrealized_conversion_cast %a : i8 to i1
      %c = builtin.unrealized_conversion_cast %b : i1 to i16
      return %c : i16
    })");
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));
  EXPECT_EQ(count(*module, "builtin.unrealized_conversion_cast"), 0u);
  auto func = cast<func::FuncOp>(module->getBody()->front());
  auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  EXPECT_EQ(ret.getOperand(0), func.getArgument(0));
}

TEST_F(DirectConversionTest, KeepsACastChainWithAUserOfAnotherType) {
  auto module = parse(R"(
    func.func @f(%x: i16) -> i1 {
      %a = builtin.unrealized_conversion_cast %x : i16 to i8
      %b = builtin.unrealized_conversion_cast %a : i8 to i1
      %c = builtin.unrealized_conversion_cast %b : i1 to i16
      return %b : i1
    })");
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));
  EXPECT_EQ(count(*module, "builtin.unrealized_conversion_cast"), 2u);
}

// The module body is a graph region, so its casts can form a cycle.
TEST_F(DirectConversionTest, ErasesAnUnusedCastCycle) {
  auto module = parse(R"(
    module {
      %a = builtin.unrealized_conversion_cast %c : i32 to i64
      %b = builtin.unrealized_conversion_cast %a : i64 to i16
      %c = builtin.unrealized_conversion_cast %b : i16 to i32
    })");
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));
  EXPECT_EQ(count(*module, "builtin.unrealized_conversion_cast"), 0u);
}

TEST_F(ThreadedDirectConversionTest, ConvertsEveryFunctionOfAModule) {
  constexpr unsigned numFuncs = 300;
  auto module = parse(makeModuleText(
      numFuncs, "  %a = arith.constant 1 : i32\n  return %a : i32"));
  ASSERT_TRUE(module);
  EXPECT_FALSE(convert(*module));

  unsigned numWideConstants = 0;
  module->walk([&](arith::ConstantOp op) {
    numWideConstants += op.getType().isInteger(64);
  });
  EXPECT_EQ(numWideConstants, numFuncs);
  // Each `return` still wants an i32, so each function keeps one bridge cast.
  EXPECT_EQ(count(*module, "builtin.unrealized_conversion_cast"), numFuncs);
}

TEST_F(ThreadedDirectConversionTest, ReportsTheFirstUnconvertedOp) {
  auto module = parse(
      makeModuleText(300, "  %y = arith.muli %x, %x : i32\n  return %y : i32"));
  ASSERT_TRUE(module);

  SmallVector<unsigned> errorLines;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    if (auto loc = dyn_cast<FileLineColLoc>(diag.getLocation()))
      errorLines.push_back(loc.getLine());
    return success();
  });
  EXPECT_TRUE(failed(applyDirectConversion(
      *module, &converter, makePatterns(), {},
      [](Operation *op) { return isa<arith::MulIOp>(op); })));
  // The `muli` of the first function is on line 2 of the module.
  EXPECT_EQ(errorLines, SmallVector<unsigned>{2});
}
