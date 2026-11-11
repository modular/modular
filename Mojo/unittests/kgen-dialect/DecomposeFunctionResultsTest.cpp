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

#include "Mojo/CODialect/CODialect.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace testing;

namespace {

using Plan = std::optional<SmallVector<DecomposedArgLeaf>>;

DecomposedArgLeaf leafAt(Type type, unsigned field) {
  DecomposedArgLeaf leaf;
  leaf.type = type;
  leaf.path.push_back(field);
  return leaf;
}

/// Decomposes every `kgen.struct` result into its fields; arguments are left
/// alone. The default policy does the opposite, so this is the only way to
/// reach the result paths of the pass.
class DecomposeStructResultsPolicy
    : public FunctionArgumentDecompositionPolicy {
public:
  FailureOr<Plan> planArgument(FuncOp, unsigned, Type) override {
    return Plan();
  }

  FailureOr<Plan> planResult(FuncOp, unsigned, Type resType) override {
    auto st = dyn_cast<StructType>(resType);
    if (!st)
      return Plan();
    std::optional<SmallVector<Type>> elts = st.getElementTypes();
    if (!elts)
      return Plan();
    SmallVector<DecomposedArgLeaf> leaves;
    for (auto [i, type] : llvm::enumerate(*elts))
      leaves.push_back(leafAt(type, static_cast<unsigned>(i)));
    return Plan(std::move(leaves));
  }
};

/// Keeps one field of a struct result and drops the rest, which no rebuild can
/// close: the leaves do not cover the whole result.
class DecomposeFirstFieldResultPolicy
    : public FunctionArgumentDecompositionPolicy {
public:
  FailureOr<Plan> planArgument(FuncOp, unsigned, Type) override {
    return Plan();
  }

  FailureOr<Plan> planResult(FuncOp, unsigned, Type resType) override {
    auto st = dyn_cast<StructType>(resType);
    if (!st)
      return Plan();
    std::optional<SmallVector<Type>> elts = st.getElementTypes();
    if (!elts || elts->empty())
      return Plan();
    return Plan(SmallVector<DecomposedArgLeaf>{leafAt(elts->front(), 0)});
  }
};

/// Strips the pointer off a field the argument carries, so the function takes
/// the pointee and a whole-aggregate use has to be rebuilt through a slot.
class StripPointerFieldPolicy : public FunctionArgumentDecompositionPolicy {
public:
  FailureOr<Plan> planArgument(FuncOp, unsigned, Type argType) override {
    auto st = dyn_cast<StructType>(argType);
    if (!st)
      return Plan();
    std::optional<SmallVector<Type>> elts = st.getElementTypes();
    if (!elts || elts->size() != 1)
      return Plan();
    auto ptr = dyn_cast<PointerType>((*elts)[0]);
    if (!ptr)
      return Plan();
    return Plan(
        SmallVector<DecomposedArgLeaf>{leafAt(ptr.getElementType(), 0)});
  }

  FailureOr<Plan> planResult(FuncOp, unsigned, Type) override { return Plan(); }
};

class DecomposeFunctionResultsTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  OwningOpRef<ModuleOp> module;

  DecomposeFunctionResultsTest() {
    ctx.loadDialect<KGENDialect>();
    ctx.loadDialect<HLCF::HLCFDialect>();
    ctx.loadDialect<POP::POPDialect>();
  }

  bool parse(StringRef ir) {
    ScopedDiagnosticHandler handler(
        &ctx, [&](Diagnostic &diag) { ADD_FAILURE() << diag.str(); });
    module = parseSourceString<ModuleOp>(ir, &ctx);
    return static_cast<bool>(module);
  }

  LogicalResult
  run(std::unique_ptr<FunctionArgumentDecompositionPolicy> policy) {
    PassManager pm(&ctx);
    pm.addPass(createDecomposeFunctionArguments(std::move(policy)));
    return pm.run(*module);
  }
};

FuncOp funcNamed(ModuleOp module, StringRef name) {
  FuncOp found;
  module.walk([&](FuncOp f) {
    if (f.getSymName() == name)
      found = f;
  });
  return found;
}

CallOp singleCall(ModuleOp module) {
  CallOp found;
  module.walk([&](CallOp c) { found = c; });
  return found;
}

/// The callee returns its fields separately, and the caller re-forms the
/// struct so its old use of the whole result keeps working.
TEST_F(DecomposeFunctionResultsTest, RebuildsADecomposedStructResult) {
  ASSERT_TRUE(parse(R"mlir(
kgen.func @callee(%a: memref<128xf32>, %b: memref<128xf32>) -> !kgen.struct<(memref<128xf32>, memref<128xf32>)> {
  %0 = kgen.struct.create(%a, %b) : !kgen.struct<(memref<128xf32>, memref<128xf32>)>
  hlcf.return %0 : !kgen.struct<(memref<128xf32>, memref<128xf32>)>
}

kgen.func @caller(%a: memref<128xf32>, %b: memref<128xf32>) {
  %slot = pop.stack_allocation 1 x memref<128xf32>
  %0 = kgen.call @callee(%a, %b) : (memref<128xf32>, memref<128xf32>) -> !kgen.struct<(memref<128xf32>, memref<128xf32>)>
  %1 = kgen.struct.extract %0[0] : <(memref<128xf32>, memref<128xf32>)>
  pop.store %1, %slot : !kgen.pointer<memref<128xf32>>
  hlcf.return
}
    )mlir"));
  ASSERT_TRUE(succeeded(run(std::make_unique<DecomposeStructResultsPolicy>())));

  FuncOp callee = funcNamed(*module, "callee");
  ASSERT_TRUE(callee);
  EXPECT_EQ(callee.getFunctionType().getNumResults(), 2);

  // The caller's call now yields the two leaves, and the field-0 use reads
  // the struct rebuilt from them.
  CallOp call = singleCall(*module);
  ASSERT_TRUE(call);
  ASSERT_EQ(call.getNumResults(), 2);

  StructCreateOp rebuilt;
  call->getParentOfType<FuncOp>().walk(
      [&](StructCreateOp create) { rebuilt = create; });
  ASSERT_TRUE(rebuilt);
  EXPECT_EQ(rebuilt->getNumOperands(), 2);
  EXPECT_EQ(rebuilt->getOperand(0), call.getResult(0));
  EXPECT_EQ(rebuilt->getOperand(1), call.getResult(1));

  StructExtractOp use;
  module->walk([&](StructExtractOp extract) { use = extract; });
  ASSERT_TRUE(use);
  EXPECT_EQ(use.getContainer(), rebuilt.getResult());
}

/// A plan that leaves part of a result uncovered cannot be rebuilt. The pass
/// reports it before moving any use, so the caller is not left half-rewritten.
TEST_F(DecomposeFunctionResultsTest, IncompleteCoverageLeavesUsesAlone) {
  ASSERT_TRUE(parse(R"mlir(
kgen.func @callee(%a: memref<128xf32>, %b: memref<128xf32>) -> !kgen.struct<(memref<128xf32>, memref<128xf32>)> {
  %0 = kgen.struct.create(%a, %b) : !kgen.struct<(memref<128xf32>, memref<128xf32>)>
  hlcf.return %0 : !kgen.struct<(memref<128xf32>, memref<128xf32>)>
}

kgen.func @caller(%a: memref<128xf32>, %b: memref<128xf32>) {
  %0 = kgen.call @callee(%a, %b) : (memref<128xf32>, memref<128xf32>) -> !kgen.struct<(memref<128xf32>, memref<128xf32>)>
  %1 = kgen.struct.extract %0[0] : <(memref<128xf32>, memref<128xf32>)>
  %slot = pop.stack_allocation 1 x memref<128xf32>
  pop.store %1, %slot : !kgen.pointer<memref<128xf32>>
  hlcf.return
}
    )mlir"));

  CallOp call = singleCall(*module);
  ASSERT_TRUE(call);
  Value original = call.getResult(0);
  StructExtractOp use;
  module->walk([&](StructExtractOp extract) { use = extract; });
  ASSERT_TRUE(use);

  std::vector<std::string> errors;
  {
    ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Error)
        errors.push_back(diag.str());
    });
    EXPECT_TRUE(
        failed(run(std::make_unique<DecomposeFirstFieldResultPolicy>())));
  }

  ASSERT_FALSE(errors.empty());
  EXPECT_NE(errors.front().find("cannot rebuild a decomposed result"),
            std::string::npos);
  EXPECT_EQ(use.getContainer(), original);
}

/// A whole-aggregate use of an argument whose pointer the policy stripped is
/// rebuilt in a slot typed like the original pointer, not as a fresh
/// `pointer<T, 0>`.
TEST_F(DecomposeFunctionResultsTest, RebuildsAStrippedPointerInATypedSlot) {
  ASSERT_TRUE(parse(R"mlir(
kgen.func @spill(%arg0: !kgen.struct<(pointer<memref<128xf32>>)>) {
  %slot = pop.stack_allocation 1 x !kgen.struct<(pointer<memref<128xf32>>)>
  pop.store %arg0, %slot : !kgen.pointer<!kgen.struct<(pointer<memref<128xf32>>)>>
  hlcf.return
}
    )mlir"));
  ASSERT_TRUE(succeeded(run(std::make_unique<StripPointerFieldPolicy>())));

  FuncOp func = funcNamed(*module, "spill");
  ASSERT_TRUE(func);
  ASSERT_EQ(func.getNumArguments(), 1u);
  Type argType = func.getArgumentTypes().front();
  ASSERT_TRUE(isa<MemRefType>(argType));

  // The leaf is spilled into a slot holding the pointee, and that slot is the
  // pointer the rebuilt struct carries.
  POP::StoreOp spill;
  module->walk([&](POP::StoreOp op) {
    if (op.getArg() == func.getArgument(0))
      spill = op;
  });
  ASSERT_TRUE(spill);
  auto slot = spill.getPtr().getDefiningOp<POP::StackAllocationOp>();
  ASSERT_TRUE(slot);
  EXPECT_EQ(slot.getResult().getType(), PointerType::get(argType));
}

/// Decomposing a result through a nested function-like region is not
/// supported: the region's returns describe the region, so the pass refuses
/// instead of rewriting them with this function's leaves. `co.execute` is the
/// op that has such a region today.
TEST_F(DecomposeFunctionResultsTest, RefusesANestedFunctionLikeRegion) {
  // Only this test parses a `co.execute`, so only it needs the dialect.
  ctx.loadDialect<CO::CODialect>();
  ASSERT_TRUE(parse(R"mlir(
kgen.func @f(%a: memref<128xf32>) -> !kgen.struct<(memref<128xf32>, memref<128xf32>)> {
  %h = co.execute : i32 {
    %c = kgen.param.constant: i32 = <1>
    hlcf.return %c : i32
  }
  %s = kgen.struct.create(%a, %a) : !kgen.struct<(memref<128xf32>, memref<128xf32>)>
  hlcf.return %s : !kgen.struct<(memref<128xf32>, memref<128xf32>)>
}
    )mlir"));

  std::vector<std::string> errors;
  {
    ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Error)
        errors.push_back(diag.str());
    });
    EXPECT_TRUE(failed(run(std::make_unique<DecomposeStructResultsPolicy>())));
  }
  ASSERT_FALSE(errors.empty());
  EXPECT_NE(errors.front().find("nested function-like region"),
            std::string::npos);
}

} // namespace
