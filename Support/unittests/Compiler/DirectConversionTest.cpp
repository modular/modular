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
  DirectConversionTest()
      : ctx(MLIRContext::Threading::DISABLED),
        converter(makeWideningConverter()) {
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
