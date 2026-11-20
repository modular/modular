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

#include "Mojo/Support/PrecompiledMLIR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

#include <optional>
#include <string>

using namespace mlir;
using M::KGEN::PrecompiledMLIRCache;
using M::KGEN::PrecompiledMLIRRegistry;

namespace {

constexpr PrecompiledMLIRRegistry::Kind kAddI8I8 = 1;

// The snippet the registry serves: an `i8` addition. Snippets are modules
// containing the named function, stored as textual MLIR.
constexpr llvm::StringLiteral kAddI8Source = R"mlir(
  module {
    llvm.func @add_i8(%x: i8, %y: i8) -> i8 {
      %0 = llvm.add %x, %y : i8
      llvm.return %0 : i8
    }
  }
)mlir";

// A context with the dialects the snippet + test IR need.
struct TestContext {
  MLIRContext ctx;
  TestContext() {
    ctx.loadDialect<LLVM::LLVMDialect, func::FuncDialect>();
    // The inline test uses an unregistered placeholder op.
    ctx.allowUnregisteredDialects();
  }
};

unsigned countSymbol(ModuleOp module, StringRef name) {
  unsigned n = 0;
  module.walk([&](SymbolOpInterface sym) { n += (sym.getName() == name); });
  return n;
}

//===----------------------------------------------------------------------===//

TEST(PrecompiledMLIRTest, getFunctionParsesIntoCallerStorage) {
  TestContext tc;

  PrecompiledMLIRRegistry registry({{kAddI8I8, "add_i8", kAddI8Source}});

  OwningOpRef<Operation *> storage;
  Operation *fn = registry.getFunction(kAddI8I8, &tc.ctx, storage);
  ASSERT_NE(fn, nullptr);
  EXPECT_TRUE(isa<LLVM::LLVMFuncOp>(fn));
  EXPECT_EQ(SymbolTable::getSymbolName(fn).getValue(), "add_i8");
  // The parse is owned by the caller's `storage`, not retained by the registry.
  EXPECT_TRUE(static_cast<bool>(storage));

  // Unknown kinds resolve to null rather than crashing.
  OwningOpRef<Operation *> unused;
  EXPECT_EQ(nullptr, registry.getFunction(/*kind=*/42, &tc.ctx, unused));
}

TEST(PrecompiledMLIRTest, insertIntoModuleAddsSymbolOnceThenDedups) {
  TestContext tc;

  PrecompiledMLIRRegistry registry({{kAddI8I8, "add_i8", kAddI8Source}});

  OwningOpRef<ModuleOp> target(ModuleOp::create(UnknownLoc::get(&tc.ctx)));

  FailureOr<Operation *> first = registry.insertIntoModule(kAddI8I8, *target);
  ASSERT_FALSE(failed(first));
  EXPECT_NE(*first, nullptr);
  EXPECT_EQ(
      SymbolTable::lookupSymbolIn(*target, StringAttr::get(&tc.ctx, "add_i8")),
      *first);
  EXPECT_EQ(countSymbol(*target, "add_i8"), 1u);

  // A second insert is deduplicated: same op, still a single definition.
  FailureOr<Operation *> second = registry.insertIntoModule(kAddI8I8, *target);
  ASSERT_FALSE(failed(second));
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(countSymbol(*target, "add_i8"), 1u);
}

TEST(PrecompiledMLIRTest, inlineInPlaceReplacesOpWithBody) {
  TestContext tc;

  PrecompiledMLIRRegistry registry({{kAddI8I8, "add_i8", kAddI8Source}});

  // A caller whose `test.placeholder` should be replaced by the snippet body.
  constexpr llvm::StringLiteral kCaller = R"mlir(
    func.func @caller(%a: i8, %b: i8) -> i8 {
      %r = "test.placeholder"(%a, %b) : (i8, i8) -> i8
      return %r : i8
    }
  )mlir";
  OwningOpRef<ModuleOp> caller =
      parseSourceString<ModuleOp>(kCaller, ParserConfig(&tc.ctx));
  ASSERT_TRUE(caller);

  Operation *placeholder = nullptr;
  caller->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "test.placeholder")
      placeholder = op;
  });
  ASSERT_NE(placeholder, nullptr);

  IRRewriter rewriter(&tc.ctx);
  ASSERT_FALSE(failed(registry.inlineInPlace(
      kAddI8I8, placeholder, placeholder->getOperands(), rewriter)));

  // The placeholder is gone and an `llvm.add` took its place.
  bool sawPlaceholder = false;
  LLVM::AddOp add;
  caller->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "test.placeholder")
      sawPlaceholder = true;
    if (auto a = dyn_cast<LLVM::AddOp>(op))
      add = a;
  });
  EXPECT_FALSE(sawPlaceholder);
  ASSERT_TRUE(add);

  // The function's return now yields the inlined add's result, computed from
  // the caller's own arguments.
  func::FuncOp fn = cast<func::FuncOp>(
      SymbolTable::lookupSymbolIn(*caller, StringAttr::get(&tc.ctx, "caller")));
  Operation *ret = fn.getBody().front().getTerminator();
  ASSERT_EQ(ret->getNumOperands(), 1u);
  EXPECT_EQ(ret->getOperand(0), add.getResult());
  EXPECT_EQ(add.getLhs(), fn.getArgument(0));
  EXPECT_EQ(add.getRhs(), fn.getArgument(1));
}

TEST(PrecompiledMLIRTest, cacheParsesOncePerKindAndRegistry) {
  TestContext tc;

  PrecompiledMLIRRegistry first({{kAddI8I8, "add_i8", kAddI8Source}});
  PrecompiledMLIRRegistry second({{kAddI8I8, "add_i8", kAddI8Source}});

  PrecompiledMLIRCache cache(&tc.ctx);
  Operation *fn = cache.getFunction(first, kAddI8I8);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(SymbolTable::getSymbolName(fn).getValue(), "add_i8");

  // Repeated lookups reuse the parse rather than producing a fresh one.
  EXPECT_EQ(cache.getFunction(first, kAddI8I8), fn);

  // Kind numbering is per-registry, so two registries must not collide.
  EXPECT_NE(cache.getFunction(second, kAddI8I8), fn);

  // Unknown kinds resolve to null, and stay null on the second lookup.
  EXPECT_EQ(cache.getFunction(first, /*kind=*/42), nullptr);
  EXPECT_EQ(cache.getFunction(first, /*kind=*/42), nullptr);
}

TEST(PrecompiledMLIRTest, registryCopiesSnippetStrings) {
  TestContext tc;

  // The registry owns its copies, so snippets assembled from temporaries stay
  // usable once those are gone.
  std::optional<PrecompiledMLIRRegistry> registry;
  {
    std::string symbolName = "add_i8";
    std::string mlirText = kAddI8Source.str();
    registry.emplace(
        PrecompiledMLIRRegistry::Snippet{kAddI8I8, symbolName, mlirText});
  }

  OwningOpRef<Operation *> storage;
  Operation *fn = registry->getFunction(kAddI8I8, &tc.ctx, storage);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(SymbolTable::getSymbolName(fn).getValue(), "add_i8");
}

} // namespace
