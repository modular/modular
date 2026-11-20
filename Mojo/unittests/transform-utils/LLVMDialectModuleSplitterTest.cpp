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

#include "Mojo/TransformUtils/SlicingUtils.h"
#include "Mojo/TransformUtils/SplitPlanner.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

using namespace M;
using namespace mlir;
using namespace testing;

namespace {

using SliceNames = std::vector<std::vector<std::string>>;

bool noSliceLocalGlobals(LLVM::GlobalOp) { return false; }

class LLVMDialectModuleSplitterTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  std::string errors;
  /// The slices of the last `split`, and the functions it outlined.
  std::vector<OwningOpRef<ModuleOp>> sliceModules;
  std::vector<std::string> outlined;

  LLVMDialectModuleSplitterTest() { ctx.loadDialect<LLVM::LLVMDialect>(); }

  /// Splits `source` and returns the names of the top-level ops of each slice,
  /// in slice order. Returns nothing when `plan` fails, with its diagnostics
  /// in `errors`. The source is not verified, so it can call an undefined
  /// function.
  std::optional<SliceNames>
  split(StringRef source,
        function_ref<bool(LLVM::GlobalOp)> isSliceLocal = noSliceLocalGlobals,
        unsigned outlineThreads = 0) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        source, ParserConfig(&ctx, /*verifyAfterParse=*/false));
    if (!module) {
      ADD_FAILURE() << "cannot parse the test input";
      return std::nullopt;
    }

    ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
      errors += diag.str() + "\n";
      return llvm::success();
    });
    LLVMDialectModuleSplitter splitter(std::move(module));
    if (failed(splitter.plan(isSliceLocal, outlineThreads)))
      return std::nullopt;
    for (StringAttr name : splitter.getOutlinedFunctions())
      outlined.push_back(name.str());

    SliceNames names;
    for (size_t idx = 0, e = splitter.getNumSlices(); idx != e; ++idx) {
      OwningOpRef<ModuleOp> &slice =
          sliceModules.emplace_back(splitter.takeSlice(idx));
      EXPECT_TRUE(succeeded(verify(*slice))) << "slice " << idx;
      std::vector<std::string> &sliceNames = names.emplace_back();
      for (Operation &op : *slice->getBody()) {
        auto symbol = dyn_cast<SymbolOpInterface>(op);
        sliceNames.push_back(symbol ? symbol.getName().str()
                                    : op.getName().getStringRef().str());
      }
    }
    return names;
  }

  LLVM::LLVMFuncOp lookupFunc(size_t slice, StringRef name) {
    return sliceModules[slice]->lookupSymbol<LLVM::LLVMFuncOp>(name);
  }
};

TEST_F(LLVMDialectModuleSplitterTest, ClonesSharedFunctionIntoEachSlice) {
  std::optional<SliceNames> slices = split(R"mlir(
    llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4>]
    llvm.func internal @shared() {
      llvm.return
    }
    llvm.func internal @only_a() {
      llvm.return
    }
    llvm.func @kernel_a() {
      llvm.call @shared() : () -> ()
      llvm.call @only_a() : () -> ()
      llvm.return
    }
    llvm.func @kernel_b() {
      llvm.call @shared() : () -> ()
      llvm.return
    }
  )mlir");
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(*slices,
            (SliceNames{{"llvm.module_flags", "shared", "only_a", "kernel_a"},
                        {"llvm.module_flags", "shared", "kernel_b"}}));
}

constexpr const char *kKernelsSharingAGlobal = R"mlir(
  llvm.mlir.global internal @counter(0 : i32) : i32
  llvm.func @kernel_a() {
    %0 = llvm.mlir.addressof @counter : !llvm.ptr
    llvm.return
  }
  llvm.func @kernel_b() {
    %0 = llvm.mlir.addressof @counter : !llvm.ptr
    llvm.return
  }
  llvm.func @kernel_c() {
    llvm.return
  }
)mlir";

TEST_F(LLVMDialectModuleSplitterTest, KeepsUsersOfMutableGlobalTogether) {
  std::optional<SliceNames> slices = split(kKernelsSharingAGlobal);
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(*slices,
            (SliceNames{{"counter", "kernel_a", "kernel_b"}, {"kernel_c"}}));
}

TEST_F(LLVMDialectModuleSplitterTest, ClonesSliceLocalGlobalIntoEachUser) {
  std::optional<SliceNames> slices =
      split(kKernelsSharingAGlobal, [](LLVM::GlobalOp global) {
        return global.getSymName() == "counter";
      });
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(*slices, (SliceNames{{"counter", "kernel_a"},
                                 {"counter", "kernel_b"},
                                 {"kernel_c"}}));
}

TEST_F(LLVMDialectModuleSplitterTest, KeepsExportedCalleeWithItsCaller) {
  std::optional<SliceNames> slices = split(R"mlir(
    llvm.func @helper() {
      llvm.return
    }
    llvm.func @kernel_a() {
      llvm.call @helper() : () -> ()
      llvm.return
    }
    llvm.func @kernel_b() {
      llvm.return
    }
  )mlir");
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(*slices, (SliceNames{{"helper", "kernel_a"}, {"kernel_b"}}));
}

constexpr const char *kKernelsSharingAFunction = R"mlir(
  llvm.func internal @shared(%arg0: i32) -> i32 {
    %0 = llvm.add %arg0, %arg0 : i32
    %1 = llvm.mul %0, %0 : i32
    llvm.return %1 : i32
  }
  llvm.func @kernel_a(%arg0: i32) -> i32 {
    %0 = llvm.call @shared(%arg0) : (i32) -> i32
    llvm.return %0 : i32
  }
  llvm.func @kernel_b(%arg0: i32) -> i32 {
    %0 = llvm.call @shared(%arg0) : (i32) -> i32
    llvm.return %0 : i32
  }
)mlir";

TEST_F(LLVMDialectModuleSplitterTest, OutlinesSharedFunction) {
  std::optional<SliceNames> slices =
      split(kKernelsSharingAFunction, noSliceLocalGlobals, /*threads=*/2);
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(
      *slices,
      (SliceNames{{"shared"}, {"shared", "kernel_a"}, {"shared", "kernel_b"}}));
  EXPECT_EQ(outlined, std::vector<std::string>{"shared"});

  LLVM::LLVMFuncOp definition = lookupFunc(0, "shared");
  EXPECT_FALSE(definition.isExternal());
  EXPECT_EQ(definition.getLinkage(), LLVM::Linkage::WeakODR);
  EXPECT_TRUE(definition.getDsoLocal());
  for (size_t slice : {1, 2}) {
    LLVM::LLVMFuncOp declaration = lookupFunc(slice, "shared");
    EXPECT_TRUE(declaration.isExternal()) << "slice " << slice;
    EXPECT_EQ(declaration.getLinkage(), LLVM::Linkage::External);
    EXPECT_FALSE(declaration.getDsoLocal());
  }
}

TEST_F(LLVMDialectModuleSplitterTest, DropsSubprogramFromDeclaration) {
  std::optional<SliceNames> slices = split(R"mlir(
    #file = #llvm.di_file<"shared.mojo" in "">
    #cu = #llvm.di_compile_unit<
      id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #file,
      isOptimized = true, emissionKind = Full>
    #sp = #llvm.di_subprogram<
      id = distinct[1]<>, compileUnit = #cu, scope = #file, name = "shared",
      file = #file, subprogramFlags = Definition>
    llvm.func internal @shared(%arg0: i32) -> i32 {
      %0 = llvm.add %arg0, %arg0 : i32
      %1 = llvm.mul %0, %0 : i32
      llvm.return %1 : i32
    } loc(fused<#sp>["shared.mojo":1:1])
    llvm.func @kernel_a(%arg0: i32) -> i32 {
      %0 = llvm.call @shared(%arg0) : (i32) -> i32
      llvm.return %0 : i32
    }
    llvm.func @kernel_b(%arg0: i32) -> i32 {
      %0 = llvm.call @shared(%arg0) : (i32) -> i32
      llvm.return %0 : i32
    }
  )mlir",
                                           noSliceLocalGlobals, /*threads=*/2);
  ASSERT_TRUE(slices) << errors;
  ASSERT_EQ(slices->size(), 3u);

  auto hasSubprogram = [](LLVM::LLVMFuncOp func) {
    return isa<FusedLocWith<LLVM::DISubprogramAttr>>(func.getLoc());
  };
  EXPECT_TRUE(hasSubprogram(lookupFunc(0, "shared")));
  for (size_t slice : {1, 2})
    EXPECT_FALSE(hasSubprogram(lookupFunc(slice, "shared")))
        << "slice " << slice;
}

TEST_F(LLVMDialectModuleSplitterTest, ClonesAlwaysInlineSharedFunction) {
  std::string source = kKernelsSharingAFunction;
  StringRef signature = "-> i32 {";
  source.replace(source.find(signature), signature.size(),
                 "-> i32 attributes {always_inline} {");
  std::optional<SliceNames> slices =
      split(source, noSliceLocalGlobals, /*threads=*/2);
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(*slices,
            (SliceNames{{"shared", "kernel_a"}, {"shared", "kernel_b"}}));
  EXPECT_TRUE(outlined.empty());
}

TEST_F(LLVMDialectModuleSplitterTest, IgnoresUnresolvedDiscardableReference) {
  std::optional<SliceNames> slices = split(R"mlir(
    llvm.func @kernel_a() attributes {test.ref = @elsewhere} {
      llvm.return
    }
    llvm.func @kernel_b() {
      llvm.return
    }
  )mlir");
  ASSERT_TRUE(slices) << errors;
  EXPECT_EQ(*slices, (SliceNames{{"kernel_a"}, {"kernel_b"}}));
}

TEST_F(LLVMDialectModuleSplitterTest, RejectsComdat) {
  EXPECT_FALSE(split(R"mlir(
    llvm.comdat @__llvm_comdat {
      llvm.comdat_selector @any any
    }
    llvm.func @kernel_a() {
      llvm.return
    }
    llvm.func @kernel_b() {
      llvm.return
    }
  )mlir"));
  EXPECT_NE(errors.find("'llvm.comdat' op is not supported when splitting a "
                        "module per exported function"),
            std::string::npos)
      << errors;
}

TEST_F(LLVMDialectModuleSplitterTest, RejectsConstructorList) {
  EXPECT_FALSE(split(R"mlir(
    llvm.func @init() {
      llvm.return
    }
    llvm.mlir.global_ctors ctors = [@init], priorities = [0 : i32], data = [#llvm.zero]
    llvm.func @kernel_a() {
      llvm.return
    }
  )mlir"));
  EXPECT_NE(errors.find("'llvm.mlir.global_ctors' op names a symbol, which is "
                        "not supported when splitting a module per exported "
                        "function"),
            std::string::npos)
      << errors;
}

TEST_F(LLVMDialectModuleSplitterTest, RejectsCallToUndefinedFunction) {
  EXPECT_FALSE(split(R"mlir(
    llvm.func @kernel_a() {
      llvm.call @missing() : () -> ()
      llvm.return
    }
    llvm.func @kernel_b() {
      llvm.return
    }
  )mlir"));
  EXPECT_NE(errors.find("'llvm.call' op names 'missing', which is not defined "
                        "in the module"),
            std::string::npos)
      << errors;
}

using Slices = std::vector<std::vector<unsigned>>;

SplitSymbol makeFunction(SmallVector<unsigned, 4> dependencies) {
  SplitSymbol symbol;
  symbol.dependencies = std::move(dependencies);
  symbol.isFunction = true;
  symbol.isAnchor = true;
  return symbol;
}

SplitSymbol makeMutableGlobal() {
  SplitSymbol symbol;
  symbol.tiesUsers = true;
  return symbol;
}

TEST(PlanSplitPerFunctionTest, KeepsUserOfMutableGlobalWithItsCaller) {
  // 1 is a leaf, 2 uses the global 0, and 3 calls both.
  std::vector<SplitSymbol> symbols = {makeMutableGlobal(), makeFunction({}),
                                      makeFunction({0}), makeFunction({1, 2})};
  SplitPlan plan = planSplitPerFunction(symbols);
  EXPECT_EQ(plan.slices, (Slices{{1}, {0, 2, 3}}));
  EXPECT_TRUE(plan.outlineRoots.empty());
}

TEST(PlanSplitPerFunctionTest, GivesRecursiveUserOfMutableGlobalItsOwnSlice) {
  // 1 uses the global 0 and calls itself.
  std::vector<SplitSymbol> symbols = {makeMutableGlobal(), makeFunction({0, 1}),
                                      makeFunction({})};
  EXPECT_EQ(planSplitPerFunction(symbols).slices, (Slices{{0, 1}, {2}}));
}

} // namespace
